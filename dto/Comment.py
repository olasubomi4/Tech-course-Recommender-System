import pandas as pd
class Comment:
    def __init__(self):
        self.id = None
        self.content=None
        self.rating=None
        self.created_at=None
        self.autor_id=None
        self.autor_name=None
        self.autor_initials=None
        self.course_id=None


    def get_id(self):
        return self.id

    def get_content(self):
        return self.content

    def get_rating(self):
        return self.rating
    def get_created_at(self):
        return self.created_at

    def get_autor_id(self):
        return self.autor_id

    def get_autor_name(self):
        return self.autor_name

    def get_autor_initials(self):
        return self.autor_initials

    def get_course_id(self):
        return self.course_id

    def set_id(self, id):
        self.id = id

    def set_content(self, content):
        self.content = content

    def set_rating(self, rating):
        self.rating = rating

    def set_created_at(self, created_at):
        self.created_at = created_at

    def set_autor_id(self, autor_id):
        self.autor_id = autor_id

    def set_autor_name(self, autor_name):
        self.autor_name = autor_name

    def set_autor_initials(self, autor_initials):
        self.autor_initials = autor_initials

    def set_course_id(self, course_id):
        self.course_id = course_id


    def convertCommentToDataFrame(self):
        comment = vars(self)
        comment = pd.DataFrame([comment])
        return comment
