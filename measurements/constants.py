import util.io.env


BASE_DIR_ENV_NAME = 'MEASUREMENTS_DIR'
BASE_DIR = util.io.env.load(BASE_DIR_ENV_NAME)

# earth
EARTH_RADIUS = 6371 * 10**3
MAX_SEA_DEPTH = 11 * 10**3
