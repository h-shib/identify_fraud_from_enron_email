import matplotlib.pyplot as plt

def show_plot(num_of_features, precision, recall, algo_name):
    plt.plot(num_of_features, precision, marker='o', label='Precision')
    plt.plot(num_of_features, recall, marker='o', label='Recall')
    plt.title('Precision and Recall vs. Number of Features for ' + algo_name)
    plt.xlabel('K Best Features')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('scores_' + algo_name + '.png')
    plt.show()