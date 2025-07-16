!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="hVps0XPzHyrftbIqVfpn")
project = rf.workspace("sagar-0blpa").project("rock_paper_scissor-lgln5")
version = project.version(2)
dataset = version.download("folder")
                