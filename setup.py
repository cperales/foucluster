from setuptools import setup

setup(
    name='foucluster',
    description='Clustering of songs using Fourier Transform',
    long_description='Similarities among songs are computed using Fast Fourier '
                'Transform. With this information, unsupervised machine learning'
                 ' is applied.',
    url='https://github.com/cperales/foucluster',
    version='1.4',
    author='Carlos Perales',
    author_email='sir.perales@gmail.com',
    keywords=['cluster', 'Fourier', 'music', 'song',
              'machine learning', 'kmeans'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX :: Linux',
        'Topic :: Artistic Software',
        'Topic :: Scientific/Engineering',
        'Topic :: Multimedia :: Sound/Audio :: Analysis'
    ],
    packages=['foucluster'
              ],
    zip_safe=False,
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'sklearn',
                      'seaborn'
                      ],
    include_package_data=True,
    setup_requires=[],
    tests_require=['pytest',
                   'pytest-cov'],
    extras_require={
        'docs': [
            'sphinx'
        ]
    },
)

