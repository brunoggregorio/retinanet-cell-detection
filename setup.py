import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'mynet_keras.utils.compute_overlap',
        ['mynet_keras/utils/compute_overlap.pyx']
    ),
]


setuptools.setup(
    name             = 'retinanet-cell-detection',
    version          = '0.0.1',
    description      = 'Keras implementation of RetineNet for cell detection.',
    url              = 'https://github.com/brunoggregorio/retinanet-cell-detection',
    author           = 'Bruno Gregorio',
    author_email     = 'brunoggregorio@gmail.com',
    maintainer       = 'Bruno Gregorio',
    maintainer_email = 'brunoggregorio@gmail.com',
    cmdclass         = {'build_ext': BuildExtension},
    packages         = setuptools.find_packages(),
    install_requires = ['keras', 'keras-resnet', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python', 'progressbar2'],
    entry_points     = {
        'console_scripts': [
            'mynet_keras-train=mynet_keras.bin.train:main',
            'mynet_keras-evaluate=mynet_keras.bin.evaluate:main',
            'mynet_keras-debug=mynet_keras.bin.debug:main',
            'mynet_keras-convert-model=mynet_keras.bin.convert_model:main',
        ],
    },
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)
