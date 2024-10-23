# Description: This file contains the implementation of the Trayis AI.
# Copyright (c) 2024 Harlik Karmetes
# This code is licensed under MIT license (see LICENSE for details)

import random
import numpy as np
import multiprocessing as mp

pieceScore = {"K": 3, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1}

knightScores = [
    [-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50],
    [-0.40, -0.20,  0.00,  0.00,  0.00,  0.00, -0.20, -0.40],
    [-0.30,  0.00,  0.10,  0.15,  0.15,  0.10,  0.00, -0.30],
    [-0.30,  0.05,  0.15,  0.20,  0.20,  0.15,  0.05, -0.30],
    [-0.30,  0.00,  0.15,  0.20,  0.20,  0.15,  0.00, -0.30],
    [-0.30,  0.05,  0.10,  0.15,  0.15,  0.10,  0.05, -0.30],
    [-0.40, -0.20,  0.00,  0.05,  0.05,  0.00, -0.20, -0.40],
    [-0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50]
]

bishopScores = [
    [-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20],
    [-0.10,  0.05,  0.00,  0.00,  0.00,  0.00,  0.05, -0.10],
    [-0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10, -0.10],
    [-0.10,  0.00,  0.10,  0.10,  0.10,  0.10,  0.00, -0.10],
    [-0.10,  0.05,  0.10,  0.10,  0.10,  0.10,  0.05, -0.10],
    [-0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10, -0.10],
    [-0.10,  0.05,  0.00,  0.00,  0.00,  0.00,  0.05, -0.10],
    [-0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20]
]

queenScores = [
    [-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20],
    [-0.10,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.10],
    [-0.10,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10],
    [-0.05,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05],
    [ 0.00,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05],
    [-0.10,  0.05,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10],
    [-0.10,  0.00,  0.05,  0.00,  0.00,  0.00,  0.00, -0.10],
    [-0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20]
]

whiterookScores = [[4, 3, 4, 4, 4, 4, 3, 4],
                   [4, 4, 4, 4, 4, 4, 4, 4],
                   [1, 1, 2, 3, 3, 2, 1, 1],
                   [1, 2, 3, 4, 4, 3, 2, 1],
                   [1, 2, 3, 4, 4, 3, 2, 1],
                   [1, 1, 2, 2, 2, 2, 1, 1],
                   [3, 3, 3, 3, 3, 3, 3, 3],
                   [4, 2, 2, 2, 2, 2, 2, 4]]

blackrookScores = [[4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4],
                   [3, 3, 3, 3, 3, 3, 3, 3],
                   [1, 1, 2, 3, 3, 2, 1, 1],
                   [1, 2, 3, 4, 4, 3, 2, 1],
                   [1, 2, 3, 4, 4, 3, 2, 1],
                   [1, 1, 2, 2, 2, 2, 1, 1],
                   [4, 4, 4, 4, 4, 4, 4, 4],
                   [4, 3, 2, 1, 1, 2, 3, 4]]

whitePawnScores = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
    [0.10, 0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.10],
    [0.05, 0.05, 0.10, 0.25, 0.25, 0.10, 0.05, 0.05],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.01, 0.01, 0.03, 0.05, 0.05, 0.03, 0.01, 0.01],
    [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]
whitePawnScoresEndgame = [
    [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.30, 0.30, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]

blackPawnScores = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
    [0.01, 0.01, 0.03, 0.05, 0.05, 0.03, 0.01, 0.01],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.05, 0.05, 0.10, 0.25, 0.25, 0.10, 0.05, 0.05],
    [0.10, 0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.10],
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]
blackPawnScoresEndgame = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.30, 0.30, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
]

whitekingScores = [
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
    [-0.20, -0.30, -0.30, -0.40, -0.40, -0.30, -0.30, -0.20],
    [-0.10, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.10],
    [0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20],
    [0.20, 0.30, 0.10, 0.00, 0.00, 0.10, 0.30, 0.20]
]
whitekingScoresEndgame = [
    [-0.50, -0.40, -0.30, -0.20, -0.20, -0.30, -0.40, -0.50],
    [-0.30, -0.20, -0.10,  0.00,  0.00, -0.10, -0.20, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.30,  0.00,  0.00,  0.00,  0.00, -0.30, -0.30],
    [-0.50, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30, -0.50]
]
blackkingScores = [[0.20, 0.30, 0.10, 0.00, 0.00, 0.10, 0.30, 0.20],
                    [0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.20, 0.20],
                    [-0.10, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.10],
                    [-0.20, -0.30, -0.30, -0.40, -0.40, -0.30, -0.30, -0.20],
                    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
                    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
                    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30],
                    [-0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30]]

blackkingScoresEndgame = [
    [-0.50, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30, -0.50],
    [-0.30, -0.10,  0.00,  0.00,  0.00,  0.00, -0.10, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.30,  0.40,  0.40,  0.30, -0.10, -0.30],
    [-0.30, -0.10,  0.20,  0.30,  0.30,  0.20, -0.10, -0.30],
    [-0.30, -0.20, -0.10,  0.00,  0.00, -0.10, -0.20, -0.30],
    [-0.50, -0.40, -0.30, -0.20, -0.20, -0.30, -0.40, -0.50]
]         

piecePositionScores = {
    "N": knightScores,
    "B": bishopScores,
    "Q": queenScores,
    "wR": whiterookScores,
    "bR": blackrookScores,
    "wp": whitePawnScores,
    "bp": blackPawnScores,
    "wK": whitekingScores,
    "bK": blackkingScores

}

CHECKMATE = 10000
STALEMATE = 0
DEPTH = 4
SET_WHITE_AS_BOT = -1


# Convert position score tables to NumPy arrays
whitePawnScores = np.array(whitePawnScores)
blackPawnScores = np.array(blackPawnScores)
whitekingScores = np.array(whitekingScores)
blackkingScores = np.array(blackkingScores)
whitePawnScoresEndgame = np.array(whitePawnScoresEndgame)
blackPawnScoresEndgame = np.array(blackPawnScoresEndgame)
whitekingScoresEndgame = np.array(whitekingScoresEndgame)
blackkingScoresEndgame = np.array(blackkingScoresEndgame)

piecePositionScores = {
    "N": np.array(knightScores),
    "B": np.array(bishopScores),
    "Q": np.array(queenScores),
    "wR": np.array(whiterookScores),
    "bR": np.array(blackrookScores),
    "wp": whitePawnScores,
    "bp": blackPawnScores,
    "wK": whitekingScores,
    "bK": blackkingScores
}
#elo 1400
def findRandomMoves(validMoves):
    return validMoves[np.random.choice(len(validMoves))]
# Sample dictionary of openings (you can expand this with more openings)
openings = {
    "e2e4 e7e5": "e2e4",  # Example: King's Pawn Opening
    "d2d4 d7d5": "d2d4",  # Example: Queen's Pawn Opening
    # Add more openings as needed
}
def findBestMove(gs, validMoves, returnQueue):
    global nextMove
    nextMove = None
    np.random.shuffle(validMoves)

    if gs.playerWantsToPlayAsBlack:
        # Swap the variables
        whitePawnScores, blackPawnScores = blackPawnScores, whitePawnScores

    SET_WHITE_AS_BOT = 1 if gs.whiteToMove else -1

    findMoveNegaMaxAlphaBeta(gs, validMoves, DEPTH, -CHECKMATE, CHECKMATE, SET_WHITE_AS_BOT)

    returnQueue.put(nextMove)

# Move order part
def move_order(gs, validMoves):
    validMoves.sort(key=lambda move: gs.board[move.endRow][move.endCol] != "--", reverse=True)
    return validMoves

transposition_table = {}

def quiescenceSearch(gs, alpha, beta, turnMultiplier, depth):
    if depth == 0:
        return turnMultiplier * scoreBoard(gs)

    stand_pat = turnMultiplier * scoreBoard(gs)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    validMoves = gs.getValidMoves()
    validMoves = move_order(gs, validMoves)

    for move in validMoves:
        if gs.board[move.endRow][move.endCol] != "--":  # Only consider captures
            gs.makeMove(move)
            score = -quiescenceSearch(gs, -beta, -alpha, -turnMultiplier, depth - 1)
            gs.undoMove()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha

def findMoveNegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, turnMultiplier):
    global nextMove
    board_hash = hash(str(gs.board))
    if board_hash in transposition_table and transposition_table[board_hash][1] >= depth:
        return transposition_table[board_hash][0]

    if depth == 0:
        return quiescenceSearch(gs, alpha, beta, turnMultiplier, 3)  # Limit quiescence search to 2 moves

    maxScore = -CHECKMATE

    # Move order part
    validMoves = move_order(gs, validMoves)

    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()  # opponent validmoves
        score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth-1, -beta, -alpha, -turnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
                print(move, score)
        gs.undoMove()
        if maxScore > alpha:
            alpha = maxScore  # alpha is the new max
        if alpha >= beta:  # if we find new max is greater than minimum so far in a branch then we stop iterating in that branch as we found a worse move in that branch
            break

    transposition_table[board_hash] = (maxScore, depth)
    return maxScore

def endgame_phase_weight(material_count_without_pawns):
    """
    Calculate the endgame phase weight based on the material count without pawns.
    """
    endgame_material_start =  pieceScore["R"] * 2 + pieceScore["B"] + 2 * pieceScore["N"]
    multiplier = 1 / endgame_material_start
    return 1 - min(1, material_count_without_pawns * multiplier)

def blend_scores(normal_score, endgame_score, endgame_value):
    """
    Blend the scores from the normal and endgame evaluation tables based on the endgame value.
    """
    return (1 - endgame_value) * normal_score + endgame_value * endgame_score

def scoreBoard(gs):
    if gs.checkmate:
        if gs.whiteToMove:
            gs.checkmate = False
            return -CHECKMATE  # black wins
        else:
            gs.checkmate = False
            return CHECKMATE  # white wins
    elif gs.stalemate:
        return STALEMATE

    board_array = np.array(gs.board)
    white_pieces = board_array[np.char.startswith(board_array, 'w')]
    black_pieces = board_array[np.char.startswith(board_array, 'b')]

    white_materialnonepawn = np.sum([pieceScore[piece[1]] for piece in white_pieces if piece[1] != 'p'])
    black_materialnonepawn = np.sum([pieceScore[piece[1]] for piece in black_pieces if piece[1] != 'p'])
    whitematerial = np.sum(np.vectorize(lambda piece: pieceScore[piece[1]])(white_pieces))
    blackmaterial = np.sum(np.vectorize(lambda piece: pieceScore[piece[1]])(black_pieces))

    whiteendgamephaseweight = endgame_phase_weight(white_materialnonepawn)
    blackendgamephaseweight = endgame_phase_weight(black_materialnonepawn)
    endgame_value = whiteendgamephaseweight - blackendgamephaseweight



    #new code V
    # Initialize scores
    piecePositionScores = np.zeros_like(board_array, dtype=float)
    piecePositionEndgameScores = np.zeros_like(board_array, dtype=float)

    # Calculate position scores
    white_pawn_mask = board_array == "wp"
    black_pawn_mask = board_array == "bp"
    white_king_mask = board_array == "wK"
    black_king_mask = board_array == "bK"

    piecePositionScores[white_pawn_mask] = whitePawnScores[white_pawn_mask]
    piecePositionScores[black_pawn_mask] = blackPawnScores[black_pawn_mask]
    piecePositionScores[white_king_mask] = whitekingScores[white_king_mask]
    piecePositionScores[black_king_mask] = blackkingScores[black_king_mask]

    piecePositionEndgameScores[white_pawn_mask] = whitePawnScoresEndgame[white_pawn_mask]
    piecePositionEndgameScores[black_pawn_mask] = blackPawnScoresEndgame[black_pawn_mask]
    piecePositionEndgameScores[white_king_mask] = whitekingScoresEndgame[white_king_mask]
    piecePositionEndgameScores[black_king_mask] = blackkingScoresEndgame[black_king_mask]

    # Blend scores
    blendedScores = np.where(np.char.startswith(board_array, 'w'),
                             blend_scores(piecePositionScores, piecePositionEndgameScores, whiteendgamephaseweight),
                             blend_scores(piecePositionScores, piecePositionEndgameScores, blackendgamephaseweight))
    # # Calculate final score
    # score = 0
    # for row in range(len(gs.board)):
    #     for col in range(len(gs.board[row])):
    #         square = gs.board[row][col]
    #         if square != "--":
    #             if SET_WHITE_AS_BOT:
    #                 if square[0] == 'w':
    #                     score += pieceScore[square[1]] + blendedScores[row, col] * .1
    #                 elif square[0] == 'b':
    #                     score -= pieceScore[square[1]] + blendedScores[row, col] * .1
    #             else:
    #                 if square[0] == 'w':
    #                     score -= pieceScore[square[1]] + blendedScores[row, col] * .1
    #                 elif square[0] == 'b':
    #                     score += pieceScore[square[1]] + blendedScores[row, col] * .1
    # Calculate final score
    white_mask = np.char.startswith(board_array, 'w')
    black_mask = np.char.startswith(board_array, 'b')

    def get_piece_scores(mask, blended_scores):
        pieces = board_array[mask]
        piece_types =  np.array([piece[1] for piece in pieces]) # finally this stupid copilot is shitty dumbasing
        piece_scores = np.vectorize(pieceScore.get)(piece_types)
        return piece_scores + blended_scores[mask] * 0.1

    if SET_WHITE_AS_BOT:
        score = np.sum(get_piece_scores(white_mask, blendedScores)) - np.sum(get_piece_scores(black_mask, blendedScores))
    else:
        score = np.sum(get_piece_scores(black_mask, blendedScores)) - np.sum(get_piece_scores(white_mask, blendedScores))



     #new code ^
    # score = 0
    # for row in range(len(gs.board)):
    #     for col in range(len(gs.board[row])):
    #         square = gs.board[row][col]
    #         if square != "--":
    #             piecePositionScore = 0
    #             piecePositionEndgameScore = 0
    #             if square[1] == "p":
    #                 piecePositionScore = whitePawnScores[row, col] if square[0] == 'w' else blackPawnScores[row, col]
    #                 piecePositionEndgameScore = whitePawnScoresEndgame[row, col] if square[0] == 'w' else blackPawnScoresEndgame[row, col]
    #             elif square[1] == "K":
    #                 piecePositionScore = whitekingScores[row, col] if square[0] == 'w' else blackkingScores[row, col]
    #                 piecePositionEndgameScore = whitekingScoresEndgame[row, col] if square[0] == 'w' else blackkingScoresEndgame[row, col]

    #             blendedScore = blend_scores(piecePositionScore, piecePositionEndgameScore, whiteendgamephaseweight if square[0] == 'w' else blackendgamephaseweight)

    #             if SET_WHITE_AS_BOT:
    #                 if square[0] == 'w':
    #                     score += pieceScore[square[1]] + blendedScore * .1
    #                 elif square[0] == 'b':
    #                     score -= pieceScore[square[1]] + blendedScore * .1
    #             else:
    #                 if square[0] == 'w':
    #                     score -= pieceScore[square[1]] + blendedScore * .1
    #                 elif square[0] == 'b':
    #                     score += pieceScore[square[1]] + blendedScore * .1
    # Evaluate passed pawns
    if endgame_value > 0.6:
        passed_pawns = passed_Pawn(gs)
        for pawn in passed_pawns:
            row, col = pawn
            if gs.board[row][col][0] == 'w':
                score += 0.4  # Add bonus for white passed pawn
            else:
                score -= 0.4 # Add bonus for black passed pawn
    isolated_pawns = isolated_Pawn(gs)
    for pawn in isolated_pawns:
        row, col = pawn
        if gs.board[row][col][0] == 'w':
            score -= 0.2  # Subtract penalty for white isolated pawn
        else:
            score += 0.2 # Subtract penalty for black isolated pawn
    # Evaluate pawn promotion
    white_promotion_mask = (board_array == "wp") & (np.arange(len(gs.board))[:, None] == 0)
    black_promotion_mask = (board_array == "bp") & (np.arange(len(gs.board))[:, None] == 7)

    score += np.sum(white_promotion_mask) * pieceScore["Q"]  # Add bonus for white pawn promotion
    score -= np.sum(black_promotion_mask) * pieceScore["Q"]  # Add bonus for black pawn promotion


    if endgame_value > 0.5:  # Apply only in the endgame phase
        whiteKingPos = None
        blackKingPos = None
        for row in range(len(gs.board)):
            for col in range(len(gs.board[row])):
                if gs.board[row][col] == "wK":
                    whiteKingPos = (row, col)
                elif gs.board[row][col] == "bK":
                    blackKingPos = (row, col)

        if whiteKingPos and blackKingPos:
            if gs.whiteToMove:
                score += ForceKingToCornerEndgameEval(whiteKingPos, blackKingPos, endgame_value, whitematerial, blackmaterial)
            else:
                score -= ForceKingToCornerEndgameEval(blackKingPos, whiteKingPos, endgame_value, blackmaterial, whitematerial)

    return score

def ForceKingToCornerEndgameEval(friendliykingSquare = (0, 0), enemyKingSquare = (0, 0), endgameeval = 0.1, kingmaterial=6, enemykingmaterial=6):
    evaluation = 0
    if kingmaterial > enemykingmaterial + pieceScore["p"] * 2 and endgameeval > 0:
        enemykingrank = enemyKingSquare[0]
        enemykingfile = enemyKingSquare[1]

        enemykingdisttocenterfile = np.max([3 - enemykingfile, enemykingfile - 4])
        enemykingdisttocenterrank = np.max([3 - enemykingrank, enemykingrank - 4])

        enemykingdisttocenter = enemykingdisttocenterfile + enemykingdisttocenterrank
        evaluation += enemykingdisttocenter

        friendliykingrank = friendliykingSquare[0]
        friendliykingfile = friendliykingSquare[1]

        distbetweenkingsfile = np.abs(friendliykingfile - enemykingfile)
        distbetweenkingsrank = np.abs(friendliykingrank - enemykingrank)

        distbetweenkings = distbetweenkingsfile + distbetweenkingsrank
        evaluation += 14 - distbetweenkings
        return evaluation * endgameeval *0.8
    return 0


def passed_Pawn(gs):
    passed_pawns = []
    board_array = np.array(gs.board)

    # Identify positions of white and black pawns
    white_pawn_positions = np.argwhere(board_array == "wp")
    black_pawn_positions = np.argwhere(board_array == "bp")

    # Check for white passed pawns
    for pos in white_pawn_positions:
        row, col = pos
        if row == 0:
            continue
        left_file = col - 1 if col > 0 else col
        right_file = col + 1 if col < 7 else col
        if not np.any(board_array[:row, [left_file, col, right_file]] == "bp"):
            passed_pawns.append((row, col))

    # Check for black passed pawns
    for pos in black_pawn_positions:
        row, col = pos
        if row == 7:
            continue
        left_file = col - 1 if col > 0 else col
        right_file = col + 1 if col < 7 else col
        if not np.any(board_array[row+1:, [left_file, col, right_file]] == "wp"):
            passed_pawns.append((row, col))

    return passed_pawns

def isolated_Pawn(gs):
    board_array = np.array(gs.board)
    isolated_pawns = []

    # White isolated pawns
    white_pawns = np.argwhere(board_array == "wp")
    for row, col in white_pawns:
        files = [max(col - 1, 0), min(col + 1, 7)]
        if not np.any(board_array[:, files] == "wp"):
            isolated_pawns.append((row, col))

    # Black isolated pawns
    black_pawns = np.argwhere(board_array == "bp")
    for row, col in black_pawns:
        files = [max(col - 1, 0), min(col + 1, 7)]
        if not np.any(board_array[:, files] == "bp"):
            isolated_pawns.append((row, col))

    return isolated_pawns




'''def findBestMove(gs, validMoves):
    turnMultiplier = 1 if gs.whiteToMove else -1
    opponentMinMaxScore = CHECKMATE # for bot worst score
    bestMoveForPlayer = None # for black
    random.shuffle(validMoves)
    for playerMove in validMoves:
        gs.makeMove(playerMove) # bot (black) makes a move
        opponentsMoves = gs.getValidMoves() # player (white) get all valid moves 
        opponentMaxScore = -CHECKMATE # player(opponent/white) worst possibility
        for opponentsMove in opponentsMoves:
            # the more positive the score the better the score for player(opponent)
            # player (opponent/white) makes a move for bot (black)
            gs.makeMove(opponentsMove) # player makes a move
            if gs.checkmate:
                score = -turnMultiplier * CHECKMATE # if player (white) makes a move and it results in checkmate than its the max score for player but worst for bot
            elif gs.stalemate:
                score = STALEMATE
            else:
                score = -turnMultiplier * scoreMaterial(gs.board)
            if score > opponentMaxScore:
                opponentMaxScore = score
            gs.undoMove()
        if opponentMaxScore < opponentMinMaxScore: # if player (opponent/white) moves does not result in checkmate(worst score for bot)
            ''''''
            opponentMaxScore = max score for the opponent if bot played playerMove

            it is calculating all possibles moves for player after bot makes move and store the minimum score of player after making player move in opponentMinMaxScore
            then again it check what if bot whould have played different move
            ''''''
            opponentMinMaxScore = opponentMaxScore
            bestMoveForPlayer = playerMove
        gs.undoMove()
    return bestMoveForPlayer '''

'''def findMoveMinMax(gs, validMoves, depth, whiteToMove): #depth represent how many moves ahead we want to look to find current best move
    global nextMove
    if depth == 0:
        return scoreMaterial(gs.board)
    
    if whiteToMove:
        maxScore = -CHECKMATE # worst score for white
        for move in validMoves:
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs, nextMoves, depth - 1, False)
            if score > maxScore:
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()
        return maxScore

    else:
        minScore = CHECKMATE # worst score for black
        for move in validMoves:
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs, nextMoves, depth - 1, True)
            if score < minScore:
                minScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()
        return minScore'''
# without alpha beta pruning
'''def findMoveNegaMax(gs, validMoves, depth, turnMultiplier):
    global nextMove
    if depth == 0:
        return turnMultiplier * scoreBoard(gs)
    
    maxScore = -CHECKMATE
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves() # opponent validmoves
        ''''''
        - sign because what ever opponents best score is, is worst score for us
        negative turnMultiplier because it changes turns after moves made 
        ''''''
        score = -findMoveNegaMax(gs, nextMoves, depth - 1, -turnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
        gs.undoMove()
    return maxScore'''

# calculate score of the board based on position
'''
def scoreMaterial(board):
    score = 0
    for row in board:
        for square in row:
            if square[0] == 'w':
                score += pieceScore[square[1]]
            elif square[0] == 'b':
                score -= pieceScore[square[1]]

    return score
'''
