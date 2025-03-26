class Course:
    def __init__(self):
        self.id=None
        self.category=None
        self.processed=False


    def get_id(self):
        return self.id

    def get_category(self):
        return self.category

    def get_processed(self):
        return self.processed

    def set_id(self, id):
        self.id=id

    def set_category(self, category):
        self.category=category

    def set_processed(self, processed):
        self.processed=processed
