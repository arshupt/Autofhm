from setuptools import setup, find_packages


setup(name='Friday',
      version='1.0',
      description='Python Automated Machine Learning',
      packages=find_packages(),
      classifiers=[
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8'
      ],
      install_requires=open('requirements.txt').readlines(),
      python_requires='>=3.7, <4',
      keywords='feature engineering data science machine learning hyperparameter optimization',
)