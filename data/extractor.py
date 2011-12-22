import logging
import requests
import simplejson
import time

NEW_POSTS = 'http://www.reddit.com/r/all/new/.json?sort=new'

class Extractor:
    def __init__(self, outfile):
        self.seen = set()
        self.outfile = outfile
        self.interval = 120

    def run(self):
        self.init()
        while True:
            data = self.get_new_posts()
            if data:
                self.process_posts(data)
            time.sleep(self.interval)

    def init(self):
        fh = open(self.outfile, 'r')
        self.seen.update(simplejson.loads(x)['id'] for x in fh)
        fh.close()
        logging.info("Read %d IDs", len(self.seen))
        self.writer = open(self.outfile, 'a')

    def get_new_posts(self):
        try:
            request = requests.get(NEW_POSTS)
            return simplejson.loads(request.content)
        except Exception, e:
            logging.exception("Request failed")
            return None

    def process_posts(self, data):
        if 'data' not in data or 'children' not in data['data']:
            logging.warn("Malformed data: %r", data)
            return
        try:
            num_written = 0
            for entry in data['data']['children']:
                entry_data = entry['data']
                if entry_data['id'] not in self.seen:
                    simplejson.dump(entry_data, self.writer)
                    self.writer.write('\n')
                    num_written += 1
                    self.seen.add(entry_data['id'])
            logging.info("Wrote %d new entries", num_written)
        except Exception, e:
            logging.exception("Error parsing posts")