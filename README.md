## Pseudorandom permutation generator.

Given a large set of integers which I wanted to visit in pseudorandom order,
and an acute awareness that no realistic PRNG has enough state to reach even a
small fraction of the possible permutations, I set about looking for a means to
generate random permutations via format preserving encryption (but without the
high cost of being cryptographically secure).

There are plenty of trivial PRNGs (eg., LFSR) which are simple to configure for
an arbitrary number of bits, but the number of different permutations that they
can archieve is extremely poor.  I want something with a decent key schedule,
but not something so complicated that it has to use the term "key schedule".

So... to visit [0,n) in random order, without having to enumerate them in an
array so that that array can be shuffled, we simply count from 0 to n-1 and
"encrypt" that count in a way that results in another number in [0,n).

Here I've stacked the following techniques:

**`do x = f(x); while (x >= n)`**, guaranteed to visit every value less than
`n` in reasonable time provided `f()` isn't stupid.

**n-bit 1:1 hash**  Using the same principle as the murmur3 mix function (shift,
eor, mul, shift, eor, mul, shift, eor), and performing parameter searches for
best avalanche for various bit widths other than 32 and 64.

**Random bit matrix multipy**  If you can invert the matrix then the operation is
bijective, and it's trivial to prove that every different matrix gives a
different permutation.  By multiplying two unit triangular matrices you reach
plenty of (but not all) invertible matrices and it's easy to show that the
randomisation maps 1:1 with the random bits consumed.  This consumes the bulk
of the parameterisation; for an n-bit permutation, this uses `n*(n-1)` bits.

**Multiply by an odd number**  Just any odd number.  They're all co-prime to
powers of two and that's all you need to keep it bijective.

**All of the above a whole bunch of times**  A widely-used cryptographic
technique, but very hard to prove that it makes perfect use of the entropy used
to configure it... but if it's good enough for cryptographers then it's good
enough for me.
