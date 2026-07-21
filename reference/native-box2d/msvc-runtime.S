.text
.globl msvc_CIcos
.type msvc_CIcos,@function
msvc_CIcos:
    /* MSVC9 emits every live Box2D cosf call immediately before a sinf call
       with the same float argument.  Compute both results once, retain the
       float-rounded sine, and return the cosine on the ordinary x87 stack.
       The key check in msvc_CIsin keeps standalone sinf calls correct.  The
       position solver repeatedly rotates static bodies by exact +0, whose
       reference results are exactly sin(+0)=+0 and cos(+0)=1. */
    fsts msvc_pair_key
    cmpl $0, msvc_pair_key
    jne .Lnonzero_cos
    fstps msvc_pair_sin
    fld1
    ret
.Lnonzero_cos:
    /* FSINCOS reports C2 and leaves its operand untouched when |x| >= 2^63.
       Snapshot validation permits every finite float, so preserve the MSVC
       FCOS-then-FSIN behavior outside FSINCOS's argument-reduction range. */
    pushl %eax
    movl msvc_pair_key, %eax
    andl $0x7fffffff, %eax
    cmpl $0x5f000000, %eax
    popl %eax
    jae .Lindependent_pair
    fsincos
    fxch %st(1)
    fstps msvc_pair_sin
    ret
.Lindependent_pair:
    fld %st(0)
    fcos
    fxch %st(1)
    fsin
    fstps msvc_pair_sin
    ret

.globl msvc_CIsin
.type msvc_CIsin,@function
msvc_CIsin:
    subl $4, %esp
    fstps (%esp)
    pushl %eax
    movl 4(%esp), %eax
    cmpl msvc_pair_key, %eax
    jne .Lstandalone_sin
    flds msvc_pair_sin
    popl %eax
    addl $4, %esp
    ret
.Lstandalone_sin:
    flds 4(%esp)
    popl %eax
    addl $4, %esp
    fsin
    ret

.globl msvc_CIsqrt
.type msvc_CIsqrt,@function
msvc_CIsqrt:
    fsqrt
    ret

.local msvc_pair_key
.local msvc_pair_sin
.bss
.align 4
msvc_pair_key:
    .zero 4
msvc_pair_sin:
    .zero 4

.section .note.GNU-stack,"",@progbits
