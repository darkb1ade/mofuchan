# Configuration file for jupyter-notebook.
c = get_config()

# ------------------------------------------------------------------------------
# NotebookApp(JupyterApp) configuration
# ------------------------------------------------------------------------------

c.NotebookApp.ip = "0.0.0.0"
c.NotebookApp.port = 8888
c.NotebookApp.allow_root = True
c.MultiKernelManager.default_kernel_name = "python3"
c.NotebookApp.open_browser = False
c.ServerApp.root_dir = "/workdir/notebook"
c.CodeCell.cm_config.autoCloseBrackets = True
# ## Hashed password to use for web authentication.
# # sets a password if PASSWORD is set in the environment
# if "PASSWORD" in os.environ:
#     c.NotebookApp.password = passwd(os.environ["PASSWORD"])
#     print(c.NotebookApp.password)
#     del os.environ["PASSWORD"]

# Disable authentication
c.NotebookApp.token = ""
c.NotebookApp.password = ""
