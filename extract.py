import simplejson, string
import Stemmer
import scipy.sparse

class SimpleVectorizer:
    def fit(self, ex_list):
        vocab_set = set(ex_list)
        #print 'Vocab set:', sorted(vocab_set)
        self.vocab = {}
        for (i, x) in enumerate(vocab_set):
            self.vocab[x] = i
        
    def transform(self, ex_list):
        nrows = len(ex_list)
        ncols = len(self.vocab)
        arr = scipy.sparse.lil_matrix((nrows, ncols))
        for (i, x) in enumerate(ex_list):
            if x in self.vocab:
                arr[i, self.vocab[x]] = 1
        return arr
        
    def fit_transform(self, ex_list):
        self.fit(ex_list)
        return self.transform(ex_list)
    
def find_top_reddits(filenames, topN):
    counts = {}
    for filename in filenames:
        fh = open(filename, 'r')
        for line in fh:
            try:
                data = simplejson.loads(line)
                post_class = data['subreddit']
                if post_class in counts:
                    counts[post_class] += 1
                else:
                    counts[post_class] = 1
            except simplejson.JSONDecodeError:
                pass

    sorted_keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return sorted_keys[5:5+topN]

def clean_char(c):
    if c >= 'a' and c <= 'z':
        return c
    return ' '
    
def load_posts(filenames, categories):
    stemmer = Stemmer.Stemmer('english')
    val_dict = {
            'target': list(), 
            'title': list(),
            'domain': list(),
            }
    for filename in filenames:
        fh = open(filename, 'r')
        for line in fh:
            try:
                data = simplejson.loads(line)
                post_class = data['subreddit']
                if post_class not in categories:
                    continue
                title_str = string.lower(data['title'])
                title_str = ''.join([ clean_char(x) for x in title_str ]) #clean string
                title_str = ' '.join([ stemmer.stemWord(x) for x in title_str.split(' ') ]) #stem words
                
                domain_str = string.lower(data['domain'])
                if domain_str.startswith('self.'):
                    domain_str = 'self'
                else:
                    domain_str = domain_str.split('.')
                    if len(domain_str) < 2:
                        domain_str = '.'.join(domain_str)
                    else:
                        domain_str = '.'.join(domain_str[-2:])
                
                val_dict['title'].append(title_str)
                val_dict['domain'].append(domain_str)
                val_dict['target'].append(categories.index(post_class))
            except simplejson.JSONDecodeError:
                pass
    return val_dict  