from numba.np.ufunc import parallel
from aiproject.battle import battle as bt


def main() -> None:
    battle_0 = bt.Battle(
        bt.MonteCarloPlayer(id=0, number_of_game_per_move=200),
        bt.MonteCarloPlayer(id=1, number_of_game_per_move=50),
        number_of_match=50,
    )
    battle_0.full_battle()
    print(battle_0.get_result_pretty_string())
    #print("\n")
    #battle_1 = bt.Battle(
    #    bt.MonteCarloPlayer(id=0, number_of_game_per_move=100, p=True),
    #    bt.MonteCarloTreeSearchPlayer(id=1, rollout=100),
    #    number_of_match=20
    #)
    #battle_1.full_battle()
    #print(battle_1.get_result_pretty_string())


if __name__ == "__main__":
    main()
