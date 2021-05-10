import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        f = True
        while f:
            f = False
            for item in dataset.iterate_once(batch_size):
                if self.get_prediction(item[0]) != nn.as_scalar(item[1]):
                    f = True
                    self.w.update(item[0],nn.as_scalar(item[1]))

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.w0 = nn.Parameter(1, 50)
        self.b0 = nn.Parameter(1, 50)
        self.w1 = nn.Parameter(50, 1)
        self.b1 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0)), self.w1), self.b1)
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"    
        while True:

            for item in dataset.iterate_once(self.batch_size):
                
                gradient_list = nn.gradients(self.get_loss(item[0],item[1]), [self.w0, self.w1, self.b0, self.b1])

                self.w0.update(gradient_list[0], -0.005)
                self.w1.update(gradient_list[1], -0.005)
                self.b0.update(gradient_list[2], -0.005)
                self.b1.update(gradient_list[3], -0.005)

            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer_count = 4
        self.learning_rate = -0.005
        self.neurons = 100
        self.batch_size = 25
        self.fw = nn.Parameter(784, self.neurons)
        self.fb = nn.Parameter(1, self.neurons)
        self.layers = [nn.Parameter(self.neurons, self.neurons) for _ in range(self.layer_count - 2)]
        self.bias = [nn.Parameter(1,self.neurons) for _ in range(self.layer_count - 2) ]
        self.lw = nn.Parameter(self.neurons, 10)
        self.lb = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        i=0
        while i <len(self.layers):
             layered = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.fw), self.fb)), self.layers[i]), self.bias[i])
             i+=1

        layered = nn.AddBias(nn.Linear(nn.ReLU(layered), self.lw), self.lb)
        return layered

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        ct = 0
        while dataset.get_validation_accuracy() < 0.97:
            ct+=1
            for item in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(item[0],item[1])
                gradient_list = nn.gradients(loss, [self.fw]+self.layers+[self.lw] + [self.fb]+self.bias+[self.lb])
               
                i=0
                while i<len(gradient_list):
                    if i == 0:
                        self.fw.update(gradient_list[0], self.learning_rate)
                    elif i == len(self.layers)+1:
                        self.lw.update(gradient_list[i], self.learning_rate)
                    elif i > 0 and i < len(self.layers)+1:
                        self.layers[i-1].update(gradient_list[i], self.learning_rate)
                    elif i == len(self.layers)+2:
                        self.fb.update(gradient_list[i], self.learning_rate)
                    elif i == len(gradient_list)-1:
                        self.lb.update(gradient_list[i], self.learning_rate)
                    else:
                        self.bias[i-len(self.layers) - 3].update(gradient_list[i], self.learning_rate)
                    i+=1

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.learning_rate = -0.008
        self.weights = nn.Parameter(self.num_chars, 200)
        self.function_bias = nn.Parameter(1,200)
        self.hidden_leaf_village = nn.Parameter(200,200)
        self.result_weight = nn.Parameter(200, 5)
        self.layer2 = nn.Parameter(200,200)
        self.bias2 = nn.Parameter(1,200)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        f= True
        for i in xs:
            if f:
                f = False
                continue
            a_bias = nn.AddBias(nn.ReLU(nn.Add(nn.Linear(i, self.weights), nn.Linear(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.weights),self.function_bias)), self.layer2), self.bias2), self.hidden_leaf_village))), self.function_bias)
        return nn.Linear(a_bias, self.result_weight)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        i=0
        while i<10:
            if i > 7:
                self.learning_rate = -0.001
            for item in dataset.iterate_once(self.batch_size):
                gradient_list = nn.gradients(self.get_loss(item[0], item[1]), [self.weights, self.function_bias, self.hidden_leaf_village, self.result_weight, self.layer2, self.bias2])
                self.weights.update(gradient_list[0], self.learning_rate)
                self.function_bias.update(gradient_list[1], self.learning_rate)
                self.hidden_leaf_village.update(gradient_list[2], self.learning_rate)
                self.result_weight.update(gradient_list[3], self.learning_rate)
                self.layer2.update(gradient_list[4], self.learning_rate)
                self.bias2.update(gradient_list[5], self.learning_rate)
            i+=1
