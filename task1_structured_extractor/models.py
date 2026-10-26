from pydantic import BaseModel, Field


class Experience(BaseModel):
    role: str = Field(min_length=1)
    company: str = Field(min_length=1)
    duration: str | None = None
    achievements: list[str] = Field(default_factory=list)


class Education(BaseModel):
    qualification: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    graduation_year: int | None = None


class ResumeProfile(BaseModel):
    full_name: str = Field(min_length=1)
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
