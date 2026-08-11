export class RestartGate {
  constructor(onPendingChange = () => {}) {
    this.onPendingChange = onPendingChange;
    this.pending = false;
  }

  async run(task) {
    if (this.pending) return false;
    this.pending = true;
    this.onPendingChange(true);
    try {
      return await task();
    } finally {
      this.pending = false;
      this.onPendingChange(false);
    }
  }
}
