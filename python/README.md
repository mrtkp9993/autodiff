Started as a toy project.

Higher order derivatives are not available yet, code needs some refactoring for it.

Usage:

```python
from autodiff.autodiff import *

x1 = Variable(2)
x2 = Variable(5)

f = x1.log() + x1 * x2 - sin(x2)
f.backward()
```

---
[<img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' alt='github' height='20'>](https://github.com/mrtkp9993)  [<img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='linkedin' height='20'>](https://www.linkedin.com/in/muratkoptur/)  [<img src='https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white' alt='website' height='20'>](https://muratkoptur.com) [<img src='https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white' alt='website' height='20'>](https://twitter.com/mrtkp9993)
---