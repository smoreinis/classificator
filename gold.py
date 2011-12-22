import logging, extract, models
import numpy as np
import scipy.sparse
from operator import itemgetter
from optparse import OptionParser
import sys, string
from time import time

from sklearn.feature_extraction.text import Vectorizer, WordNGramAnalyzer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import Bootstrap, ShuffleSplit

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--topReddits",
              action="store", type="int", dest="top_categories",
              help="Select top N subreddits")
op.add_option("--chi2_words",
              action="store", type="int", dest="words_chi2",
              help="Select some number of word features using a chi-squared test")
op.add_option("--chi2_sites",
              action="store", type="int", dest="sites_chi2",
              help="Select some number of domain features using a chi-squared test")
op.add_option("--topFeatures",
              action="store", type="int", dest="top_words",
              help="Print N most discriminative terms per class"
                   " for every classifier.")

(opts, args) = op.parse_args()
if len(args) < 1:
    print 'Need at least one file'
    sys.exit(1)

print __doc__
op.print_help()
print

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    #return s if len(s) <= 80 else s[:77] + "..."
    retStr = ''
    for (i, x) in enumerate(s):
        if (i > 1 and (i - 1) % 80 == 0):
            retStr += '\n'
        retStr += x
    return retStr
    
def merge_mat(A, B):
    arows, acols = A.shape
    brows, bcols = B.shape
    if (arows != brows):
        print '%d != %d' %(arows, brows)
        return None
        
    j = acols + bcols
    A = A.tolil().reshape((arows, j))
    for i in range(brows):
        A[i, acols:j] = A[i, acols:j] + B[i,:]
    return A


###############################################################################
# Load some categories from the training set
categories = [
    'occupywallstreet',
    'atheism',
    'politics',
]

if opts.top_categories:
    print "Loading Reddit posts for top %d categories:" %(opts.top_categories)
    categories = extract.find_top_reddits(args, opts.top_categories)
    print categories
else:
    print "Loading Reddit posts for categories:"
    print categories          

all_data = extract.load_posts(args, categories)
num_posts = len(all_data['target'])
print '%d posts loaded' % num_posts
print "%d categories" % len(categories)
print

model_list = {
    'log': [
        {'penalty': 'l1'},
        {'penalty': 'l2'},
    ],
#    'ridge': [{}],
    'sgd': [
        {'shuffle': True, 'loss': 'log', 'penalty': 'l1'},
        {'shuffle': True, 'loss': 'log', 'penalty': 'l2'},
        {'shuffle': True, 'loss': 'log', 'penalty': 'elasticnet'},
    ],
    'bnb': [{}],
    'mnb': [{}],
#    'lsvc': [
#        {'penalty': 'l2', 'loss': 'l1'},
#        {'penalty': 'l2', 'loss': 'l2'},
#    ],
#    'nsvc': [{'probability': True}],
#    'svc': [{'probability': True}],
}

# split a training set and a test set
iter = ShuffleSplit(num_posts, n_iterations=1, test_fraction=0.15, indices=False)
for (iter_no, (train_index, test_index)) in enumerate(iter):
    print 'Iteration no. %d' %(iter_no + 1)
    y_train = np.array([ x for (x, y) in zip(all_data['target'], train_index) if y ])
    y_test  = np.array([ x for (x, y) in zip(all_data['target'], test_index) if y ])
    print 'Sampled %d training and %d test posts' %(len(y_train), len(y_test))

    print "Extracting features from the training dataset using a sparse vectorizer"
    t0 = time()
    title_vectorizer = Vectorizer(
        analyzer=WordNGramAnalyzer(
            charset='utf-8', 
            stop_words=set(['a', 'an', 'and', 'in', 'is', 'of', 'on', 'the', 'to']),
            )
        )
    title_train = title_vectorizer.fit_transform([ x for (x, y) in zip(all_data['title'], train_index) if y ])
    
    domain_vectorizer = extract.SimpleVectorizer()
    domain_train = domain_vectorizer.fit_transform([ x for (x, y) in zip(all_data['domain'], train_index) if y ])
    X_train = title_train
    print "done in %fs" % (time() - t0)
    print "n_samples: %d, n_features: %d" % X_train.shape
    print

    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
    title_test = title_vectorizer.transform([ x for (x, y) in zip(all_data['title'], test_index) if y ])
    domain_test = domain_vectorizer.transform([ x for (x, y) in zip(all_data['domain'], test_index) if y ])
    X_test = domain_test
    print "done in %fs" % (time() - t0)
    print "n_samples: %d, n_features: %d" % X_test.shape
    print

    if opts.words_chi2:
        print ("Extracting %d best word features by a chi-squared test" %
               opts.words_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.words_chi2)
        title_train = ch2.fit_transform(title_train, y_train)
        title_test = ch2.transform(title_test)
        print "done in %fs" % (time() - t0)
        print
        
    if opts.sites_chi2:
        print ("Extracting %d best domain features by a chi-squared test" %
               opts.sites_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.sites_chi2)
        domain_train = ch2.fit_transform(domain_train, y_train)
        domain_test = ch2.transform(domain_test)
        print "done in %fs" % (time() - t0)
        print

    vocabulary = np.array([t for t, i in sorted(title_vectorizer.vocabulary.iteritems(),
                                                key=itemgetter(1))])
    for model_name in model_list:
        for model_params in model_list[model_name]:
            clf_model = models.make_classifier(model_name, params=model_params)
            print "Training: "
            print clf_model.clf
            clf_model.train(title_train, y_train)
            #clf_model.train((title_train, domain_train), y_train)
            print "train time: %0.3fs" % clf_model.train_time
            
            #pred_labels = clf_model.test_label((title_test, domain_test))
            pred_labels = clf_model.test_label(title_test)
            
            if opts.top_words and hasattr(clf_model.clf, 'coef_'):
                print "top %d keywords per class:" %(opts.top_words)
                for i, category in enumerate(categories):
                    top10 = np.argsort(clf_model.clf.coef_[i, :])[-1 * opts.top_words:]
                    print trim("%s: %s" % (category, " ".join(vocabulary[top10])))
                print
            
            print "confusion matrix:"
            print metrics.confusion_matrix(y_test, pred_labels)
            score = metrics.f1_score(y_test, pred_labels)
            print "f1-score:   %0.3f" % (score)
            print "classification report:"
            print metrics.classification_report(y_test, pred_labels)
