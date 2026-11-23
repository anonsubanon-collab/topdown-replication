"""The evidence rule: how the checks of one program become a number that is added to the judge's log-odds.

A check is an expectation (precision p = share of predictor samples that agree) against an observation (reliability r). A prediction error is an
observation that differs from the expectation. Per check:
    P(error | correct) = pc = 1 - p*r
    P(error | buggy)   = pb = Q*(1 - p*(1 - r)) + (1 - Q)*(1 - p*r),   Q = 0.5 (an input exposes the bug half the time)
If at least one prediction error remains, the evidence is the sum of log(pc / pb) over the erroneous checks only, so a counterexample is not
cancelled by inputs that match. If none remains, it is the mean of log((1 - pc) / (1 - pb)) over all checks. TopDown's score is the judge's
log P(YES) - log P(NO) plus this evidence; a program is accepted if the score is positive."""
import math

Q_EXPOSE = 0.5


def evidence(checks):
    terms = [(1 - c["precision"] * c["r"],
              Q_EXPOSE * (1 - c["precision"] * (1 - c["r"])) + (1 - Q_EXPOSE) * (1 - c["precision"] * c["r"]),
              c["disc"]) for c in checks]
    if not terms: return 0.0
    if any(d for _, _, d in terms): return sum(math.log(pc / pb) for pc, pb, d in terms if d)
    return sum(math.log((1 - pc) / (1 - pb)) for pc, pb, _ in terms) / len(terms)
