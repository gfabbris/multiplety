from setuptools import setup
#import versioneer

setup(name='multiplety',
      version=0.1,
      #cmdclass=versioneer.get_cmdclass(),
      description='Python based multiplet calculation of RIXS and XAS spectra',
      url='https://github.com/gfabbris/multiplety',
      author='Gilberto Fabbris',
      author_email='gfabbris@bnl.gov',
      license='MIT',
      packages=['multiplety'],
      install_requires=['matplotlib','numpy','scipy', 'pandas'],
      zip_safe=False)
