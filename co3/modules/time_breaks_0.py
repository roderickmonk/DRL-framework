import numpy as np

TIME_BREAKS = np.concatenate(
    (
        np.arange(0, 10),
        np.arange(10, 61, 5),
        np.arange(120, 601, 60),
        np.arange(1200, 3601, 600),
        np.arange(7200, 86401, 3600),
        np.arange(172800, 864001, 86400),
    )
)
