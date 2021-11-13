
class results:

    @staticmethod
    def format_percentage(num):
        return "{:.0%}".format(num )

    @staticmethod
    def CM_accuracy(cm):
        return cm.diagonal().sum() / cm.sum() 

    @staticmethod
    def accuracy(tp, tn , fp , fn ):
        return (tp + tn)/(tp+tn+fp+fn)

    @staticmethod
    def recall(tp , fn ):
        return (tp)/(tp+fn)

    @staticmethod
    def precision(tp , fp ):
        return (tp)/(tp+fp)

    @staticmethod
    def f1(tp , fp , fn ):
        return tp / (tp + (0.5 * (fp+fn)))

    @staticmethod
    def print_results(tp , tn , fp , fn ):
        print(">>> showing results:")
        print("--- accuracy: " + results.accuracy(tp , tn , fp , fn))
        print("--- precision: " + results.precision(tp , fp ))
        print("--- recall: " + results.recall(tp , fn ))
        print("--- f1: " + results.f1(tp , fp , fn ))
        configure.print_line()

    @staticmethod
    def get_results(tp , tn , fp , fn ):
        return [ results.accuracy(tp , tn , fp , fn) , results.precision(tp , fp ) , results.recall(tp , fn ) , results.f1(tp , fp , fn ) ]
