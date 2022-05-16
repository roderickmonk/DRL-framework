from pymongo.collection import Collection
from collections import deque
from pymongo.errors import BulkWriteError


class MongoBulkInsert:
    """"""

    def __init__(self, collection: Collection, bulk_insert_interval=100):

        self.buffer = deque()
        self.bulk_insert_interval = bulk_insert_interval
        self.collection = collection
        self.archived = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.bulk_insert(flush=True)

    def __add__(self, entry):

        self.archived += 1
        self.buffer.append(entry)
        self.bulk_insert()

    def bulk_insert(self, *, flush=False):

        buffer = self.buffer

        if len(buffer) > 0 and (len(buffer) == self.bulk_insert_interval or flush):

            try:

                self.collection.insert_many(buffer, ordered=False)

            except BulkWriteError as err:

                for e in err.details["writeErrors"]:
                    if e["code"] != 11000:  # ignore duplicates
                        raise RuntimeError(f"{e['errmsg']}")

            finally:
                buffer.clear()
