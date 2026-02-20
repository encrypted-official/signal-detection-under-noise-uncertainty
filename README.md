# Spectrum Sensing in Cognitive Radio Using Cyclostationary Features

## Group 11 (Wireless Communication), Section 1 (CSE400)

### Milestone 1

Initial research and understanding phase, where we selected a research paper based on estimating probabilistic uncertainty related to Wireless Orthogonal __Frequency-Division Multiplexing (ODFM)__.

### Milestone 2

We correlated 4 primary mathematical estimation models to detect noise and determine its probability functions (estimations):

- Signal Observation
- Noise Statistical
- Cyclostationary Signa
- Multi-Cycle Detection & Distribution

We selected __Binary Hypothesis Formulation__ as our primary detection driver. Whereby,

__H<sub>o</sub> : x[n] = w[n]__ \
__H<sub>1</sub> : x[n] = s[n] + w[n]__

Here, _w[n]_ represents noise, and _s[n]_ is our preferred OFDM signal.

Using these models, and their random variables, we formalize the detection problem. In theory, we expect each model to work as:

- Observational Model --> sample collection
- Cyclostationary model --> feature extraction
- Detection model --> statistical computation
- Distribution model --> threshold selection

Under these assumptions and statistics, we know that _H<sub>o</sub>_ follows F-distribtion. This will help in calculating both, Probability of Detection (P<sub>d</sub>) and Probability of False Alarm (P<sub>f</sub>).

In summary, we have created __a stronger theoretical clarity__ regarding the paper, while also __formalizing our hypothesis problem statement__.
