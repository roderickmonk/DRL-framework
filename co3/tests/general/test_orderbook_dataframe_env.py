import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import logging
import pprint
from importlib import import_module

import numpy as np
import pytest
from env_modules.continuous_action_space_env import ContinuousActionSpaceEnv as env

logging.basicConfig(
    format="[%(levelname)-5s] %(message)s", level=logging.INFO, datefmt=""
)

np.set_printoptions(precision=10, floatmode="fixed", edgeitems=25)  # type:ignore


# setattr(env, "get_reward", getattr(import_module("get_reward_profit"), "get_reward"))
setattr(env, "get_reward", import_module("get_reward.get_reward_profit"))


def test_get_reward_0():

    env.pdf_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 10
    env.pdf_y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    env.weight = np.sum(env.pdf_y)
    env.ql = 5.0
    env.precision = 4
    env.is_buy = False
    env.market_tick = 10**-env.precision

    state = np.array([0.1, 0.2, 0.4, 0.6, 0, 27, 88, 133])

    # Check rewards for a number of actions
    rewards = [
        np.around(
            env.get_reward.get_reward(  # type:ignore
                env, action=action, state=state, mid_price=0.003
            )[0],
            4,
        )
        for action in np.arange(0, 1, 0.1)
    ]

    logging.debug(f"{rewards=}")

    expected_rewards = [
        -0.6333,
        -0.2533,
        -0.4933,
        -0.4233,
        -0.6333,
        -0.6333,
        -0.6333,
        -0.6333,
        -0.6333,
        -0.6333,
    ]

    assert rewards == expected_rewards


def test_get_reward_1():

    env.pdf_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 10
    env.pdf_y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    env.weight = np.sum(env.pdf_y)
    env.ql = 5.0
    env.precision = 4
    env.is_buy = True
    env.market_tick = 10**-env.precision

    state = np.array([0.1, 0.12, 0.14, 0.16, 0, 27, 88, 133])

    rewards = [
        np.around(
            env.get_reward.get_reward(  # type:ignore
                env, action=action, state=state, mid_price=3.0
            )[
                0
            ],  # type:ignore
            4,
        )
        for action in np.arange(0, 1, 0.1)
    ]

    logging.debug(f"{rewards=}")

    expected = [
        -0.4998,
        -0.1198,
        -0.4998,
        -0.4998,
        -0.4998,
        -0.4998,
        -0.4998,
        -0.4998,
        -0.4998,
        -0.4998,
    ]

    assert rewards == expected


def _test_get_reward_2():

    pdf = np.array(
        [
            0.0005191319841484902,
            5551,
            0.0005596200715472505,
            5175,
            0.0006032659016227558,
            5917,
            0.0006503157527115049,
            5197,
            0.000701035110864253,
            4912,
            0.0007557101678920472,
            4169,
            0.0008146494362477282,
            4507,
            0.0008781854898550773,
            4319,
            0.0009466768407085534,
            4372,
            0.001020509961832575,
            5760,
            0.0011001014680152553,
            5907,
            0.0011859004666216787,
            5565,
            0.0012783910917515567,
            5814,
            0.0013780952360406677,
            6976,
            0.0014855754955206315,
            26118,
            0.0016014383441539182,
            7980,
            0.0017263375559568332,
            5842,
            0.0018609778940203569,
            5593,
            0.002006119087244747,
            6171,
            0.0021625801172271554,
            5867,
            0.002331243839491797,
            5598,
            0.0025130619651385607,
            6162,
            0.002709060431019876,
            6370,
            0.00292034518874785,
            6596,
            0.003148108445197,
            6344,
            0.003393635389715705,
            6043,
            0.003658311446005524,
            6050,
            0.003943630089588436,
            6437,
            0.004251201274973087,
            6371,
            0.004582760520071716,
            6221,
            0.004940178699128044,
            6814,
            0.005325472599414049,
            6631,
            0.0057408163012636765,
            6738,
            0.006188553445656814,
            6552,
            0.006671210458575469,
            6426,
            0.0071915108067525674,
            6446,
            0.007752390365253501,
            6066,
            0.008357013983604679,
            7116,
            0.009008793343945653,
            6076,
            0.009711406211972485,
            8537,
            0.010468817189298697,
            5422,
            0.011285300084332041,
            5758,
            0.012165462027898529,
            5511,
            0.013114269469689594,
            5110,
            0.014137076202221415,
            6770,
            0.015239653570436017,
            6647,
            0.016428223037406365,
            5945,
            0.017709491289902575,
            4905,
            0.019090688081907687,
            4832,
            0.020579607029621087,
            5247,
            0.022184649588141418,
            3933,
            0.02391487245797446,
            3750,
            0.02578003868886441,
            4114,
            0.02779067276930971,
            4400,
            0.02995812001261479,
            4702,
            0.03229461057457241,
            4650,
            0.03481332846400646,
            4147,
            0.037528485935577775,
            3655,
            0.040455403684625926,
            3876,
            0.043610597296558556,
            3260,
            0.04701187043859274,
            3075,
            0.05067841531969655,
            3540,
            0.054630920985592094,
            3299,
            0.05889169005989114,
            3332,
            0.06348476459009292,
            3506,
            0.06843606170855007,
            3334,
            0.07377351987389037,
            3153,
            0.07952725651808415,
            3109,
            0.08572973798870524,
            3252,
            0.09241596274531093,
            3695,
            0.09962365884365497,
            4953,
            0.10739349682206853,
            4548,
            0.1157693191912535,
            4640,
            0.12479838782241998,
            3318,
            0.1345316506296932,
            3341,
            0.1450240290515868,
            3882,
            0.15633472795370143,
            4241,
            0.16852756970132193,
            4258,
            0.18167135428696965,
            3606,
            0.19584024754498527,
            3157,
            0.21111419964370257,
            2234,
            0.2275793962166198,
            2133,
            0.2453287446781468,
            2031,
            0.2644623984680386,
            1896,
            0.2850883221826459,
            1761,
            0.30732290078182356,
            1712,
            0.33129159630903987,
            1368,
            0.35712965583033174,
            1148,
            0.3849828745867617,
            1185,
            0.41500841866658106,
            1093,
            0.4473757118391536,
            861,
            0.48226739155473974,
            925,
            0.519880339504514,
            635,
            0.5604267925559115,
            615,
            0.60413554033193,
            566,
            0.6512532161919086,
            450,
            0.7020456888983465,
            385,
            0.7567995628224545,
            387,
            0.8158237951535802,
            302,
            0.8794514392378597,
            275,
            0.9480415238831598,
            253,
            1.0219810792345694,
            267,
            1.1016873206517659,
            196,
            1.187610002911111,
            151,
            1.2802339580164326,
            140,
            1.3800818309384812,
            111,
            1.4877170287198893,
            96,
            1.603746899586417,
            74,
            1.7288261600031114,
            53,
            1.8636605890130693,
            38,
            2.009011010704679,
            36,
            2.1656975872790385,
            27,
            2.3346044469418326,
            23,
            2.516684672733269,
            17,
            2.7129656804463225,
            28,
            2.9245550159790135,
            13,
            3.1526466048331687,
            16,
            3.398527489023455,
            14,
            3.663585089410702,
            6,
            3.94931503443837,
            11,
            4.257329599447021,
            6,
            4.589366804186912,
            7,
            4.94730021986283,
            5,
            5.333149541048969,
            2,
            5.749091982127465,
            2,
            6.197474562556771,
            5,
            6.680827350291413,
            0,
            7.2018777380811425,
            0,
            7.7635658332057185,
            2,
            8.369061047484305,
            2,
            9.021779981171075,
            0,
            9.725405701649809,
            1,
            10.483908525710373,
            0,
            11.301568422674345,
            0,
            12.182999164782922,
            0,
            13.133174361119229,
            0,
            14.157455521965696,
            0,
            15.261622311953813,
            0,
            16.451905162714368,
            0,
            17.735020429050095,
            0,
            19.118208287004855,
            2,
        ]
    )

    pdf = pdf.reshape(-1, 2)

    logging.debug(f"{pdf=}")

    env.pdf_x = pdf[:, 0]
    logging.debug(f"{env.pdf_x=}")

    env.pdf_y = pdf[:, 1]
    logging.debug(f"{env.pdf_y=}")

    env.weight = np.sum(env.pdf_y)

    env.ql = 0.2
    env.precision = 8
    env.is_buy = False
    env.market_tick = 10**-env.precision

    import os

    from dataset import Dataset

    source = "/".join(
        [
            os.environ.get("CO3_PATH"),
            "datasets/gacsell_test_set_snt.json",
        ]  # type:ignore
    )
    df = Dataset().read(source=source)

    state = df.iloc[0][0]
    logging.critical(f"{state=}")

    mid_price = df.iloc[0][1]
    logging.critical(f"{mid_price=}")

    state = np.array(
        [
            4.061284e-04,
            4.210295e-03,
            6.068390e-03,
            6.393357e-03,
            9.002513e-03,
            9.152823e-03,
            9.850481e-03,
            1.239909e-02,
            1.332789e-02,
            1.800730e-02,
            2.153292e-02,
            2.228626e-02,
            3.107012e-02,
            7.132196e-02,
            8.755411e-02,
            8.755411e-02,
            8.755411e-02,
            8.755411e-02,
            8.755411e-02,
            8.755411e-02,
            0.000000e00,
            4.597934e-01,
            1.970570e00,
            1.580518e01,
            1.785222e01,
            2.020641e01,
            2.431665e01,
            2.934350e01,
            3.117715e01,
            3.275856e01,
            4.686416e01,
            4.952253e01,
            5.113195e01,
            5.382342e01,
            5.571145e01,
            5.571145e01,
            5.571145e01,
            5.571145e01,
            5.571145e01,
            5.571145e01,
        ]
    )

    adjusted_tick = env.market_tick / mid_price  # type: ignore

    actions, _ = np.array_split(state, 2)

    actions -= adjusted_tick

    actions = [0.49970877170562744]

    results = []

    for action in actions:
        result = env.get_reward(  # type: ignore
            env, action=action, state=state, mid_price=mid_price, detailed=True
        )
        logging.debug("\n" + pprint.pformat(result))
        results.append(result)

    import csv

    csv_file = "./rewards_trace.csv"

    try:
        with open(csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            for data in results:
                writer.writerow(data)

    except IOError:
        print("I/O error")


@pytest.mark.skip
def test_get_reward_3():

    import os

    from dataset import Dataset

    _log = logging.debug

    pdf_path = "/".join([os.environ.get("CO3_PATH"), "PDFs/sac-test2"])  # type:ignore
    pdf = np.genfromtxt(pdf_path, delimiter=",")

    _log(f"{pdf=}")

    env.pdf_x = pdf[:, 0]
    logging.debug(f"{env.pdf_x=}")

    env.pdf_y = pdf[:, 1]
    logging.debug(f"{env.pdf_y=}")

    env.weight = np.sum(env.pdf_y)

    env.ql = 0.2
    env.precision = 8
    env.is_buy = False
    env.market_tick = 10**-env.precision

    source = "/".join(
        [
            os.environ.get("CO3_PATH"),
            "datasets/gacsell_test_set_snt.json",
        ]  # type:ignore
    )
    df = Dataset().read(source=source)

    state = df.iloc[0][0]
    _log(f"{state=}")

    mid_price = df.iloc[0][1]
    _log(f"{mid_price=}")

    adjusted_tick = env.market_tick / mid_price  # type: ignore

    actions, _ = np.array_split(state, 2)  # type: ignore

    actions -= adjusted_tick

    actions = [0.49970877170562744]

    results = []

    for action in actions:
        result = env.get_reward(  # type:ignore
            env, action=action, state=state, mid_price=mid_price, detailed=True
        )
        logging.debug("\n" + pprint.pformat(result))
        results.append(result)

    import csv

    csv_file = "./rewards_trace.csv"

    try:
        with open(csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            for data in results:
                writer.writerow(data)

    except IOError:
        print("I/O error")
