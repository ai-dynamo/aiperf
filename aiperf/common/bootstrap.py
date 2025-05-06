from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.service import ServiceBase


def bootstrap_and_run_service(
    service_type: type[ServiceBase], config: ServiceConfig | None = None
):
    """Bootstrap the service and run it.

    This function will load the service configuration, create an instance of the service,
    and run it.

    Args:
        service_type: The class of the service to run
        config: The service configuration to use, if not provided, the service configuration
                will be loaded from the config file

    """
    import uvloop

    uvloop.install()

    # Load the service configuration
    if config is None:
        from aiperf.common.config.loader import load_service_config

        config = load_service_config()

    service = service_type(config=config)
    uvloop.run(service.run())
