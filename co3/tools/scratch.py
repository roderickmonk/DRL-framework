from tqdm import tqdm
from tqdm.auto import trange
from time import sleep
import time
import random
import itertools as it

MRs = [-1.0, -4.0, -5.0, -6.0, -3.4]
expected_profits = [1.5, 4.5, 5.5, 6.0, 3.4]

episodes = 100

process_start_time = time.monotonic()

child_id_counter = it.count(1)

with trange(episodes, colour="blue") as t:

    for episode in t:

        MR = random.choice(MRs)
        expected_profit = random.choice(expected_profits)

        t.set_description(f"TRAIN   {MR=:>5.1f}, Expected Profit={expected_profit}")

        sleep(2)

        if episode and episode % 5 == 0:

            child_start_time = time.monotonic()

            child_episodes = 30

            with trange(child_episodes, colour="green") as t:

                child_id = next(child_id_counter)

                MR = random.choice(MRs)
                expected_profit = random.choice(expected_profits)

                for child_episode in t:

                    t.set_description(
                        f"Test {child_id:02} "
                        f"{MR=:>5.1f}, "
                        f"Expected Profit={expected_profit}"
                    )

                    sleep(0.5)

