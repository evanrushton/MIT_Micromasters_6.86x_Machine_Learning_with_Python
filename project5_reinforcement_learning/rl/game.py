import framework
import pdb

framework.load_game_data()

DEBUG = True

while True:
    state, mission, _ = framework.newGame()
    score = 0
    print("New game start.")
    print("---")
    print(f"You find yourself in a room. {state}")

    while True:
        input_array = input(f"{mission} [{score}]>").split(" ")
        if len(input_array) == 2:
            action = input_array[0]
            obj = input_array[1]
            try:
                action_id = framework.get_actions().index(action)
                object_id = framework.get_objects().index(obj)
            except ValueError:
                action_id = -1
                object_id = -1

        else:
            action_id = -1
            object_id = -1

        state, mission, reward, terminal = framework.step_game(state, mission, action_id, object_id)
        score += reward
        print(state)

        if terminal:
            print(f"You win! Final score {score}")
            break
