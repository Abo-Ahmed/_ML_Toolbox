
class results:

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.cm = []
        self.allAcc = []
        self.allLoss = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def print_results(tp , tn , fp , fn ):
        print(">>> showing results:")
        print("--- accuracy: " + self.accuracy(tp , tn , fp , fn))
        print("--- precision: " + self.precision(tp , fp ))
        print("--- recall: " + self.recall(tp , fn ))
        print("--- f1: " + self.f1(tp , fp , fn ))
        configure.print_line()

    def calculate_results(self):
        [self.accuracy , self.precision , self.recall , self.f1] = self.get_results(self.tp , self.tn , self.fp , self.fn )
        return [self.accuracy , self.precision , self.recall , self.f1]

    def format_percentage(self, num):
        return "{:.0%}".format(num )

    def accuracy(self,tp, tn , fp , fn ):
        return (tp + tn)/(tp+tn+fp+fn)

    def accuracy_score(self , actual , predicted):
        return accuracy_score(actual, predicted)

    def recall(self,tp , fn ):
        return (tp)/(tp+fn)

    def precision(self,tp , fp ):
        return (tp)/(tp+fp)

    def f1(self,tp , fp , fn ):
        return tp / (tp + (0.5 * (fp+fn)))

    def cm_accuracy(self,cm):
        return cm.diagonal().sum() / cm.sum() 

    def plot_array(self,arr):
        plt.plot(arr)
        plt.show()
    
    def show_graphs(self):
        self.plot_array(self.allAcc)
        self.plot_array(self.allLoss)
    
    @staticmethod
    def confusion_matrix( actual , predicted ):
        data = {'y_Actual':    actual, 'y_Predicted': predicted }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
