from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class AuthorResponse(BaseModel):
    id: int
    content: Optional[str]

class User(BaseModel):
    id:  Optional[int]
    title:  Optional[str]
    name:  Optional[str]
    display_name:  Optional[str]
    image_50x50:  Optional[HttpUrl]
    initials:  Optional[str]
    public_display_name:  Optional[str]


class CourseReview(BaseModel):
    id:  Optional[int]
    content:  Optional[str]
    rating:  Optional[float]
    created:  Optional[str]
    modified:  Optional[str]
    user_modified:  Optional[str]
    user:  Optional[User]
    response: Optional[AuthorResponse]
    content_html:  Optional[str]
    created_formatted_with_time_since: Optional[str]

class UdemyResponse(BaseModel):
    count:  Optional[int]
    next: Optional[HttpUrl]
    previous: Optional[HttpUrl]
    results:  Optional[List[CourseReview]]

    def getCourseList(self) -> List[CourseReview]:
        return self.results


