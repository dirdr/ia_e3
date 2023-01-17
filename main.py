import numpy as np
from random_n import pvp_multiple_match, get_score_pretty_string

def main_pvp() -> None:
    """
    main function for ia vs ia matchs
    """
    number_of_game: int = 10
    ia100p: int = 100
    ia1000p: int = 1000
    ia10000p: int = 10000
    score_100_100: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=ia100p, nog_player_1=ia100p
    )
    score_100_1000: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=ia100p, nog_player_1=ia1000p, p=True
    )
    score_100_10000: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=ia100p, nog_player_1=ia10000p
    )
    print(get_score_pretty_string(score_100_100, number_of_game))
    print(get_score_pretty_string(score_100_1000, number_of_game))
    print(get_score_pretty_string(score_100_10000, number_of_game))


def main() -> None:
    main_pvp()


if __name__ == "__main__":
    main()
