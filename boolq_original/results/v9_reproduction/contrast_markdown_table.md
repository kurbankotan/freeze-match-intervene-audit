| Model | bal. acc O | bal. acc A | bal. acc B | edit cost | polarity cost | contrast consistency | flip rate |
|---|---|---|---|---|---|---|---|
| head | 55.13 | 68.06 | 64.12 | -12.35 | 3.94 | 34.16 | 35.8 |
| lcca | 89.83 | 94.68 | 75.81 | -3.7 | 18.87 | 75.72 | 76.13 |
| mlp | 83.03 | 91.51 | 75.28 | -7.4 | 16.23 | 68.73 | 68.73 |

Chance contrast consistency 25.0; a model answering identically to both arms scores 0.0. Polarity cost is balanced acc(A) - balanced acc(B).
