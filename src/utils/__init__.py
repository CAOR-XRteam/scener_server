# import utils.environment as env
# import utils.config as conf
# import utils.logger as logger


# config = None


# def init():
#     global config
#     if config is None:
#         env.clear_terminal()  # Only clear the terminal once
#         config = conf.load_config()  # Load configuration once
#         assert config is not None, "Config should not be None"
#         logger.configure_logger()
#         print("hello")
