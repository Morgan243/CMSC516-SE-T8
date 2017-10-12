from setuptools import setup
setup(name='CMSC516-SE-Eight',
      version='0.1',
      description='SemEval Task 8',
      author='Morgan Stuart',
      modules=['utils'],
      packages=['SemEvalEight'],#['modeling', 'data_prep'],
      requires=['numpy', 'pandas', 'keras',
                'lasagne', 'theano', 'sklearn',
                'pytorch', 'attrs'])