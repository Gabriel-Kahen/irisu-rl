"""Durable, content-addressed JSON artifacts for R3 operations."""

from __future__ import annotations

import ctypes
from contextlib import closing
import errno
import hashlib
import json
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .r3b_experiments import ExactResumeArtifact


_SHA256_LENGTH = 64
_FILE_MODE = 0o600
_DIRECTORY_MODE = 0o700
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_RECEIPT_KIND = "irisu.r3b.exact-resume-verification-receipt"
_RECEIPT_VERSION = "r3b-exact-resume-verification-receipt-v1"


class ArtifactStoreError(ValueError):
    """Base class for rejected artifact-store operations."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Stored bytes or filesystem metadata do not satisfy the store contract."""


class ArtifactTypeError(ArtifactStoreError):
    """An artifact has a different kind or schema version than requested."""


class UnsafeArtifactIdError(ArtifactStoreError):
    """An artifact identifier is not a canonical lower-case SHA-256."""


class ArtifactExistsError(ArtifactStoreError, FileExistsError):
    """A destination collision did not contain the requested artifact."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonzero_sha256(value: object) -> bool:
    return _is_sha256(value) and value != "0" * _SHA256_LENGTH


def _validate_label(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ArtifactStoreError(f"{name} must be a short nonempty UTF-8 string")
    return value


def _validate_json(value: object, *, location: str = "payload", depth: int = 0) -> None:
    if depth > 100:
        raise ArtifactStoreError(f"{location} exceeds the maximum nesting depth")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ArtifactStoreError(f"{location} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json(item, location=f"{location}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactStoreError(f"{location} contains a non-string key")
            _validate_json(item, location=f"{location}.{key}", depth=depth + 1)
        return
    raise ArtifactStoreError(
        f"{location} contains unsupported JSON value {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite JSON with one byte representation."""

    _validate_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, UnicodeError, ValueError) as error:
        raise ArtifactStoreError("value cannot be encoded as canonical JSON") from error


def ensure_private_directory(path: str | Path) -> Path:
    """Create or verify one owned, non-linked directory with mode 0700."""

    supplied = Path(path)
    if not supplied.is_absolute():
        raise ArtifactStoreError("private directory path must be absolute")
    try:
        supplied.mkdir(mode=_DIRECTORY_MODE)
    except FileExistsError:
        pass
    metadata = supplied.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise ArtifactIntegrityError(
            "private directory must be owned, non-linked, and mode 0700"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(supplied, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != _DIRECTORY_MODE
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ArtifactIntegrityError("private directory metadata changed")
    finally:
        os.close(descriptor)
    return supplied


def write_all(descriptor: int, payload: bytes) -> None:
    """Write every payload byte or fail before the caller publishes it."""

    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing private state")
        view = view[written:]


def publish_private_file(path: str | Path, payload: bytes) -> Path:
    """Atomically publish one immutable mode-0600 file without replacement."""

    destination = Path(path)
    if not destination.is_absolute() or destination.name in {"", ".", ".."}:
        raise ArtifactStoreError("private file path must be absolute and named")
    parent = ensure_private_directory(destination.parent)
    temporary = f".private-{os.getpid()}-{secrets.token_hex(16)}.tmp"
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(parent, flags)
    published = False
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        file_flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, file_flags, _FILE_MODE, dir_fd=parent_fd)
        try:
            os.fchmod(descriptor, _FILE_MODE)
            write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _rename_noreplace(
            temporary,
            destination.name,
            source_fd=parent_fd,
            destination_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
    finally:
        if not published:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return destination


def _content_sha256(*, kind: str, version: str, payload: object) -> str:
    content = {"kind": kind, "payload": payload, "version": version}
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _decode_canonical(data: bytes) -> object:
    def reject_constant(value: str) -> object:
        raise ArtifactIntegrityError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactIntegrityError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except ArtifactIntegrityError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError) as error:
        raise ArtifactIntegrityError("artifact is not valid UTF-8 JSON") from error
    try:
        canonical = canonical_json_bytes(value)
    except ArtifactStoreError as error:
        raise ArtifactIntegrityError(str(error)) from error
    if canonical != data:
        raise ArtifactIntegrityError("artifact bytes are not canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    version: str
    kind: str
    payload: object
    content_sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "kind": self.kind,
            "payload": self.payload,
            "version": self.version,
        }

    @property
    def artifact_id(self) -> str:
        return self.content_sha256


def _envelope(kind: str, version: str, payload: object) -> ArtifactEnvelope:
    kind = _validate_label(kind, name="kind")
    version = _validate_label(version, name="version")
    _validate_json(payload)
    return ArtifactEnvelope(
        version,
        kind,
        payload,
        _content_sha256(kind=kind, version=version, payload=payload),
    )


def _ensure_secure_root(root: Path) -> None:
    if not root.is_absolute():
        raise ArtifactStoreError("artifact-store root must be absolute")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=_DIRECTORY_MODE)
            except FileExistsError:
                pass
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactIntegrityError(
                f"artifact-store path contains symlink: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactIntegrityError(
                f"artifact-store path component is not a directory: {current}"
            )
    metadata = root.lstat()
    if (
        metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise ArtifactIntegrityError(
            "artifact-store root must be owned by this user with mode 0700"
        )


def _rename_noreplace(
    source: str, destination: str, *, source_fd: int, destination_fd: int
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if (
            renameat2(
                source_fd,
                os.fsencode(source),
                destination_fd,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
            == 0
        ):
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        if error_number not in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise OSError(error_number, os.strerror(error_number), destination)

    # Hard-link publication is the portable no-replace fallback. It has the same
    # atomic visibility and exclusivity properties; unlinking the temp leaves one link.
    os.link(
        source,
        destination,
        src_dir_fd=source_fd,
        dst_dir_fd=destination_fd,
        follow_symlinks=False,
    )
    os.unlink(source, dir_fd=source_fd)


class ArtifactStore:
    """A private directory of immutable ``<content-sha256>.json`` artifacts."""

    def __init__(
        self, root: str | Path, *, max_artifact_bytes: int = 16 * 1024 * 1024
    ) -> None:
        if (
            isinstance(max_artifact_bytes, bool)
            or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes <= 0
        ):
            raise ArtifactStoreError("max_artifact_bytes must be positive")
        self.root = Path(root)
        self.max_artifact_bytes = max_artifact_bytes
        _ensure_secure_root(self.root)

    def _root_fd(self) -> int:
        flags = os.O_RDONLY | os.O_CLOEXEC
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
        ):
            os.close(descriptor)
            raise ArtifactIntegrityError("artifact-store root metadata changed")
        return descriptor

    @staticmethod
    def _filename(artifact_id: str) -> str:
        if not _is_sha256(artifact_id):
            raise UnsafeArtifactIdError(
                "artifact id must be a canonical lower-case SHA-256"
            )
        return f"{artifact_id}.json"

    def path_for(self, artifact_id: str) -> Path:
        return self.root / self._filename(artifact_id)

    def publish(self, *, kind: str, version: str, payload: object) -> ArtifactEnvelope:
        envelope = _envelope(kind, version, payload)
        data = canonical_json_bytes(envelope.manifest())
        if len(data) > self.max_artifact_bytes:
            raise ArtifactStoreError("artifact exceeds max_artifact_bytes")
        destination = self._filename(envelope.artifact_id)
        temporary = f".artifact-{os.getpid()}-{secrets.token_hex(16)}.tmp"
        root_fd = self._root_fd()
        published = False
        already_exists = False
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, _FILE_MODE, dir_fd=root_fd)
            try:
                os.fchmod(descriptor, _FILE_MODE)
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while publishing artifact")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                _rename_noreplace(
                    temporary,
                    destination,
                    source_fd=root_fd,
                    destination_fd=root_fd,
                )
            except FileExistsError:
                already_exists = True
            else:
                published = True
                os.fsync(root_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(root_fd)
        if already_exists:
            existing = self.load(envelope.artifact_id)
            if existing != envelope:
                raise ArtifactExistsError(
                    f"artifact collision differs: {envelope.artifact_id}"
                )
            return existing
        if not published:
            raise ArtifactStoreError("artifact was not published")
        return envelope

    put = publish

    def _read(self, artifact_id: str) -> bytes:
        filename = self._filename(artifact_id)
        root_fd = self._root_fd()
        try:
            flags = os.O_RDONLY | os.O_CLOEXEC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(filename, flags, dir_fd=root_fd)
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise ArtifactIntegrityError(
                        "artifact path is a symlink"
                    ) from error
                raise
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                    or metadata.st_nlink != 1
                    or not 0 < metadata.st_size <= self.max_artifact_bytes
                ):
                    raise ArtifactIntegrityError("artifact file metadata is unsafe")
                chunks: list[bytes] = []
                remaining = metadata.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ArtifactIntegrityError("artifact file was truncated")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise ArtifactIntegrityError("artifact file grew while reading")
                after = os.fstat(descriptor)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ) != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                ):
                    raise ArtifactIntegrityError("artifact changed while reading")
                return b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            os.close(root_fd)

    def load(
        self,
        artifact_id: str,
        *,
        expected_kind: str | None = None,
        expected_version: str | None = None,
    ) -> ArtifactEnvelope:
        value = _decode_canonical(self._read(artifact_id))
        if not isinstance(value, Mapping) or set(value) != {
            "content_sha256",
            "kind",
            "payload",
            "version",
        }:
            raise ArtifactIntegrityError("artifact envelope schema differs")
        kind = value["kind"]
        version = value["version"]
        payload = value["payload"]
        content_sha256 = value["content_sha256"]
        try:
            envelope = _envelope(kind, version, payload)  # type: ignore[arg-type]
        except ArtifactStoreError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if (
            not _is_sha256(content_sha256)
            or content_sha256 != envelope.content_sha256
            or content_sha256 != artifact_id
        ):
            raise ArtifactIntegrityError("artifact content SHA-256 disagrees")
        if expected_kind is not None and kind != expected_kind:
            raise ArtifactTypeError(
                f"artifact kind {kind!r} does not match {expected_kind!r}"
            )
        if expected_version is not None and version != expected_version:
            raise ArtifactTypeError(
                f"artifact version {version!r} does not match {expected_version!r}"
            )
        return envelope

    def verify(
        self,
        artifact_id: str,
        *,
        expected_kind: str | None = None,
        expected_version: str | None = None,
    ) -> ArtifactEnvelope:
        return self.load(
            artifact_id,
            expected_kind=expected_kind,
            expected_version=expected_version,
        )

    def list(self) -> tuple[str, ...]:
        root_fd = self._root_fd()
        try:
            result: list[str] = []
            for name in os.listdir(root_fd):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ArtifactIntegrityError("artifact-store contains a symlink")
                if name.startswith(".artifact-") and name.endswith(".tmp"):
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
                    ):
                        raise ArtifactIntegrityError(
                            "artifact-store contains an unsafe temporary file"
                        )
                    continue
                if (
                    len(name) != _SHA256_LENGTH + len(".json")
                    or not name.endswith(".json")
                    or not _is_sha256(name[:_SHA256_LENGTH])
                ):
                    raise ArtifactIntegrityError(
                        f"artifact-store contains unexpected entry {name!r}"
                    )
                result.append(name[:_SHA256_LENGTH])
        finally:
            os.close(root_fd)
        for artifact_id in result:
            self.verify(artifact_id)
        return tuple(sorted(result))

    list_ids = list

    def verify_all(self) -> tuple[ArtifactEnvelope, ...]:
        return tuple(self.verify(artifact_id) for artifact_id in self.list())


class ArtifactLookupIndex:
    """Durable, untrusted O(1) lookup hints for immutable artifacts.

    Every returned artifact is still content-verified by ``ArtifactStore`` and
    must be semantically validated by the caller. A missing entry merely causes
    recomputation, so publication and indexing do not need a cross-file
    transaction.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.is_symlink():
            raise ArtifactIntegrityError("artifact index path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) != _DIRECTORY_MODE
        ):
            raise ArtifactIntegrityError(
                "artifact index parent must be owned with mode 0700"
            )
        create = not self.path.exists()
        if create:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _FILE_MODE,
            )
            os.close(descriptor)
        metadata = self.path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
            or metadata.st_nlink != 1
        ):
            raise ArtifactIntegrityError("artifact index file metadata is unsafe")
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS lookup("
                    "lookup_key TEXT PRIMARY KEY NOT NULL,"
                    "artifact_id TEXT NOT NULL,"
                    "kind TEXT NOT NULL,"
                    "version TEXT NOT NULL"
                    ") STRICT"
                )

    def _connect(self) -> sqlite3.Connection:
        metadata = self.path.lstat()
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
            or metadata.st_nlink != 1
        ):
            raise ArtifactIntegrityError("artifact index file metadata changed")
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def lookup(
        self,
        lookup_key: str,
        store: ArtifactStore,
        *,
        expected_kind: str,
        expected_version: str,
    ) -> ArtifactEnvelope | None:
        if not _is_nonzero_sha256(lookup_key):
            raise ArtifactStoreError("artifact lookup key must be a nonzero SHA-256")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT artifact_id,kind,version FROM lookup WHERE lookup_key=?",
                (lookup_key,),
            ).fetchone()
        if row is None:
            return None
        if row[1:] != (expected_kind, expected_version):
            raise ArtifactIntegrityError("artifact index type binding differs")
        return store.load(
            row[0],
            expected_kind=expected_kind,
            expected_version=expected_version,
        )

    def record(
        self,
        lookup_key: str,
        envelope: ArtifactEnvelope,
    ) -> None:
        if not _is_nonzero_sha256(lookup_key) or not isinstance(
            envelope, ArtifactEnvelope
        ):
            raise ArtifactStoreError("artifact lookup record is malformed")
        with closing(self._connect()) as connection:
            with connection:
                existing = connection.execute(
                    "SELECT artifact_id,kind,version FROM lookup WHERE lookup_key=?",
                    (lookup_key,),
                ).fetchone()
                expected = (
                    envelope.artifact_id,
                    envelope.kind,
                    envelope.version,
                )
                if existing is None:
                    connection.execute(
                        "INSERT INTO lookup VALUES (?,?,?,?)",
                        (lookup_key, *expected),
                    )
                elif existing != expected:
                    raise ArtifactIntegrityError(
                        "artifact lookup key already binds different content"
                    )


@dataclass(frozen=True, slots=True)
class ExactResumeVerificationReceipt:
    """Reloadable evidence receipt, deliberately not an ExactResumeArtifact token."""

    trial_manifest_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_model_sha256: str
    source_next_update_sha256: str
    restored_next_update_sha256: str
    source_after_state_sha256: str
    restored_after_state_sha256: str
    verifier_identity_sha256: str
    build_identity_sha256: str
    version: str = _RECEIPT_VERSION

    def __post_init__(self) -> None:
        hashes = (
            self.trial_manifest_sha256,
            self.checkpoint_manifest_sha256,
            self.checkpoint_model_sha256,
            self.source_next_update_sha256,
            self.restored_next_update_sha256,
            self.source_after_state_sha256,
            self.restored_after_state_sha256,
            self.verifier_identity_sha256,
            self.build_identity_sha256,
        )
        if (
            self.version != _RECEIPT_VERSION
            or any(not _is_nonzero_sha256(value) for value in hashes)
            or self.source_next_update_sha256 != self.restored_next_update_sha256
            or self.source_after_state_sha256 != self.restored_after_state_sha256
        ):
            raise ArtifactStoreError(
                "exact-resume receipt does not describe equal verified continuation"
            )

    @classmethod
    def capture(
        cls,
        artifact: ExactResumeArtifact,
        *,
        verifier_identity_sha256: str,
        build_identity_sha256: str,
    ) -> ExactResumeVerificationReceipt:
        from .r3b_experiments import ExactResumeArtifact

        if type(artifact) is not ExactResumeArtifact:
            raise ArtifactStoreError(
                "receipt capture requires a verified ExactResumeArtifact"
            )
        manifest = artifact.manifest()
        return cls(
            trial_manifest_sha256=manifest["trial_manifest_sha256"],
            checkpoint_manifest_sha256=manifest["checkpoint_manifest_sha256"],
            checkpoint_model_sha256=manifest["checkpoint_model_sha256"],
            source_next_update_sha256=manifest["source_next_update_sha256"],
            restored_next_update_sha256=manifest["restored_next_update_sha256"],
            source_after_state_sha256=manifest["source_after_state_sha256"],
            restored_after_state_sha256=manifest["restored_after_state_sha256"],
            verifier_identity_sha256=verifier_identity_sha256,
            build_identity_sha256=build_identity_sha256,
        )

    @classmethod
    def from_manifest(cls, value: object) -> ExactResumeVerificationReceipt:
        if not isinstance(value, Mapping):
            raise ArtifactStoreError("exact-resume receipt must be an object")
        expected = {
            "build_identity_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_model_sha256",
            "restored_after_state_sha256",
            "restored_next_update_sha256",
            "source_after_state_sha256",
            "source_next_update_sha256",
            "trial_manifest_sha256",
            "verifier_identity_sha256",
            "version",
        }
        if set(value) != expected:
            raise ArtifactStoreError("exact-resume receipt schema differs")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ArtifactStoreError("exact-resume receipt fields differ") from error

    def manifest(self) -> dict[str, str]:
        return {
            "build_identity_sha256": self.build_identity_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_model_sha256": self.checkpoint_model_sha256,
            "restored_after_state_sha256": self.restored_after_state_sha256,
            "restored_next_update_sha256": self.restored_next_update_sha256,
            "source_after_state_sha256": self.source_after_state_sha256,
            "source_next_update_sha256": self.source_next_update_sha256,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "verifier_identity_sha256": self.verifier_identity_sha256,
            "version": self.version,
        }

    def publish(self, store: ArtifactStore) -> ArtifactEnvelope:
        return store.publish(
            kind=_RECEIPT_KIND,
            version=self.version,
            payload=self.manifest(),
        )

    @classmethod
    def load(
        cls, store: ArtifactStore, artifact_id: str
    ) -> ExactResumeVerificationReceipt:
        envelope = store.load(
            artifact_id,
            expected_kind=_RECEIPT_KIND,
            expected_version=_RECEIPT_VERSION,
        )
        return cls.from_manifest(envelope.payload)

    @classmethod
    def load_verified_artifact(
        cls,
        store: ArtifactStore,
        artifact_id: str,
        *,
        expected_verifier_identity_sha256: str,
        expected_build_identity_sha256: str,
    ) -> ExactResumeArtifact:
        """Rehydrate proof authority only from a trusted immutable receipt."""

        receipt = cls.load(store, artifact_id)
        if (
            receipt.verifier_identity_sha256 != expected_verifier_identity_sha256
            or receipt.build_identity_sha256 != expected_build_identity_sha256
        ):
            raise ArtifactIntegrityError(
                "exact-resume receipt verifier or build identity differs"
            )
        from .r3b_experiments import _verified_exact_resume_artifact

        return _verified_exact_resume_artifact(
            receipt.trial_manifest_sha256,
            receipt.checkpoint_manifest_sha256,
            receipt.checkpoint_model_sha256,
            receipt.source_next_update_sha256,
            receipt.restored_next_update_sha256,
            receipt.source_after_state_sha256,
            receipt.restored_after_state_sha256,
        )


__all__ = [
    "ArtifactEnvelope",
    "ArtifactExistsError",
    "ArtifactIntegrityError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactTypeError",
    "ExactResumeVerificationReceipt",
    "UnsafeArtifactIdError",
    "canonical_json_bytes",
]
