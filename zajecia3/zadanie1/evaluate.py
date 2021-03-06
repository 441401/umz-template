import emails
import classifier


def get_acc(test, predictions_list):
    correct_number = len(list(filter(lambda x: x[0] == x[1], zip(
        map(lambda x: x.is_spam, test), predictions_list)))) / len(test)
    return correct_number


def get_sensivity(test, predictions_list):
    actually_spam = sum([1 for x in test if x.is_spam])
    TP = len([1 for x, y in zip(test, predictions_list)
              if (y == True and x.is_spam)])
    return TP / actually_spam


def get_specifity(test, predictions_list):
    actually_ham = sum([1 for x in test if not x.is_spam])
    TN = len([1 for x, y in zip(test, predictions_list)
              if (y == False and not x.is_spam)])
    return TN / actually_ham


def get_precision(test, predictions_list):
    all_positives = sum(predictions_list)
    TP = len([1 for x, y in zip(test, predictions_list)
            if (y == True and x.is_spam)])
    return TP / all_positives


def get_fmeas(test, predictions_list):
    precision = get_precision(test, predictions_list)
    sens = get_sensivity(test, predictions_list)
    return 2 * (precision * sens) / (precision + sens)


def evaluate(train_set, test_set, classifier):
    classifier.train(train_set)
    predictions_list = classifier.predict(test_set)
    acc = get_acc(test_set, predictions_list)
    sens = get_sensivity(test_set, predictions_list)
    spec = get_specifity(test_set, predictions_list)
    prec = get_precision(test_set, predictions_list)
    fmeas = get_fmeas(test_set, predictions_list)
    return acc, sens, spec, prec, fmeas


#def most_common_words(data_set, classifier):
#    classifier.most_common_words(data_set)


emails_list = emails.Email.emails_list
train_set = emails_list[:int(0.9 * len(emails_list))]
test_set = emails_list[int(0.9 * len(emails_list)):]
acc, sens, spec, prec, fmeas = evaluate(train_set, test_set, classifier.Bayes_laplac)#zero_rule, Bayes, Bayes_stemmed_lancester, Bayes_stemmed_porter, Bayes_stemmed_snowball, Bayes_laplac
print('accuracy:\t', acc)
print('sensivity:\t', sens)
print('specifity:\t', spec)
print('precision:\t', prec)
print('f-measure:\t', fmeas)

#most_common_words(emails_list, classifier.Bayes)


