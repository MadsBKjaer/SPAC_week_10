from tensor import Tensor, TensorShape
from collections import List

struct WordleLogic:
    var games: Int
    var useless_letters: Tensor[DType.bool]
    var useful_letters: Tensor[DType.bool]
    var correct_letters: Tensor[DType.bool]
    var done: Tensor[DType.bool]


    fn __init__(out self, games: Int, positions: Int = 5, letters: Int = 26):
        self.games = games
        self.useless_letters = Tensor[DType.bool](TensorShape(self.games, letters))
        self.useful_letters = Tensor[DType.bool](TensorShape(self.games, letters, positions))
        self.correct_letters = Tensor[DType.bool](TensorShape(self.games, positions))
        self.done = Tensor[DType.bool](TensorShape(self.games))

    fn reset_game_state(self) -> List[Tensor[DType.bool]]:
        return List[Tensor[DType.bool]](self.useless_letters, self.useful_letters, self.correct_letters, self.done)

    @staticmethod
    fn update_useless_letters(old: Tensor[DType.bool], new: Tensor[DType.bool]) -> Tensor[DType.bool]:
        return old + new

        
    


def main():
    logic = WordleLogic(2)

    var old: Tensor[DType.bool] = Tensor[DType.bool](TensorShape(2, 3))
    old[0][0] = True
    old[1][2] = True
    var new: Tensor[DType.bool] = Tensor[DType.bool](TensorShape(2, 3))
    old[1][0] = True
    old[1][2] = True
    

