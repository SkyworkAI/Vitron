
import os, sys
sys.path.append(os.path.dirname(__file__))
# print(os.path.dirname(__file__))
# print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), "ldm"))

import gligen.evaluator as evaluator
import gligen.trainer as trainer


# import gligen.ldm as ldm