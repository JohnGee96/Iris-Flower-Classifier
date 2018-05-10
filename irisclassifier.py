import sys 
from training import TrainANN

LEARNING_RATE           = 0.0004
MOMENTUM                = 0.5
# Initialize weight in the range [0, 0.01]
INIT_WEIGHT_MAG         = 0.1
# Small positive initial bias
INIT_BIAS               = 0.1
# Length of time training the NN
EPOCHS                  = 1000

# Hyper parameters that client can manipulate
HYPER_PARAMETERS = ["learning_rate", "momentum", "init_w_mag", "init_bias", "epochs"]
DEFAULT_PARAMETERS = [LEARNING_RATE, MOMENTUM, INIT_WEIGHT_MAG, INIT_BIAS, EPOCHS]

def main():
    mode = validate_program_call()
    if mode == "DEFAULT":
        hyper_param = DEFAULT_PARAMETERS
    else:
        # Ask user for hyperparameters
        args = read_input()
        hyper_param = replace_with_default(args)
    # Train the networks under the following hyperparameters
    print("\nModel is TRAINING - loss and accuracy graphs will be generated.")
    print("You can interact with the trained model after closing the graphs")
    model = TrainANN(*hyper_param)
    model.run()
    # Output graphs
    model.plot_n_save_graphs('./img/loss_and_accuracy.png')
    # Show the test set
    model.print_test_set_with_labels()
    # Ask user for input
    print("\nAbove are the test set which you use to run this model ")
    feed_input_sample(model)

def validate_program_call():
    if len(sys.argv) == 1:
        return "DEFAULT"
    elif len(sys.argv) == 2 and sys.argv[1] == "-t":
        return "TRAINING"
    else:
        print_usage()
        return None

def print_usage():
    print("\nUsage: python irisclassifier [-t] \
    \n*** -t {training mode}: Manipulate the hyperparameters that trains the classifier")
    sys.exit()

def read_input():
    print("\nENTER the LEARNING RATE: ")
    print(f"Default: {LEARNING_RATE}")
    lr = process_input(sys.stdin.readline(), float)
    print("\nENTER the MOMENTUM: ")
    print(f"Default: {MOMENTUM}")
    momentum = process_input(sys.stdin.readline(), float)
    print("\nENTER the INITIAL WEIGHT MAGNITUDE: ")
    print(f"Default: {INIT_WEIGHT_MAG}")
    init_w_mag = process_input(sys.stdin.readline(), float)
    print("\nENTER the INITIAL BIAS: ")
    print(f"Default: {INIT_BIAS}")
    init_bias = process_input(sys.stdin.readline(), float)
    print("\nENTER the number of EPOCHS: \n" \
          "(this is maximum length of time to train the ANN for. " \
          "Training can stop beforehand if reached target loss threshold or if is overfitting): ")
    print(f"Default: {EPOCHS}")
    epochs = process_input(sys.stdin.readline(), int)
    # Must return attributes in the same order as the 
    # described in HYPER_PARAMETERS 
    return [lr, momentum, init_w_mag, init_bias, epochs]

def process_input(s, type_constructor):
    s = s.replace(" ", "")
    if s == '\n':
        return None
    else:
        return type_constructor(s)

def replace_with_default(args):
    for i in range(len(HYPER_PARAMETERS)):
        if args[i] == None:
            args[i] = DEFAULT_PARAMETERS[i]
    return args

def feed_input_sample(model):
    print("\nEnter the 4 input features of the Iris flower in the format:")
    print(" 'Sepal-Length' 'Sepal-Width' 'Petal-Length' 'Petal-Width'")
    print("            EX: '6.7' '3.0' '5.2' '2.3'\n")
    print("(Ctrl+D to exit)", end=' ')
    try:
        sample = [float(x) for x in input().replace("'", '').split()]
    except EOFError:
        return
    except ValueError:
        print("Features must be separated by a space")
    prediction = model.predict_samples([sample])
    print(f"PREDICTION: {prediction}")
    feed_input_sample(model)
    
if __name__ == '__main__':
    main()
