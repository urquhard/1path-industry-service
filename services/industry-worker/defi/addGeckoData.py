import numpy as np
import pandas as pd

def addGeckoData(llama_data):

    no_gecko = llama_data[llama_data['gecko_id'].isna() == True]
    yes_gecko = llama_data[llama_data['gecko_id'].isna() == False]
        
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'apollo-dao'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'usdm'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'sithswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'taiga'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'vexchange'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'aequinox'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'alkemi-network-dao-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'annex'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'arbis-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'babylon-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'back-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'basketdao'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'bent-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'bearn-fi'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'cafeswap-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'cross-chain-bridge-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'auroraswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'component'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'cork'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'coinswap-space'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'deeplock'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'depth-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'donkey-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'dopple-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'dinosaureggs'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'ease'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'fiat-dao-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'definix'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'fuel-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'grimtoken'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'firebird-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'hyper-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'ichi'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'iftoken'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'itrust-governance-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'killswitch'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'levin'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'mars-protocol'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'minmax'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'mimas'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'mistswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'neptune-mutual'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'omm-tokens'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'oneswap-dao-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pego-network'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pala'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pantherswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'percent'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pinkswap-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pippi-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pluto-pluto'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'pandora-spirit'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'steakbank-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'secret-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'sienna'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'xsigma'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'xdollar'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'spectrum-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'sphynx-labs'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'starswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'steak-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'swampy'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'titano'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'toad-network'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'strudel-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'twindex'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'interest-protocol'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'unslashed-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'viper'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'vires-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'virtue'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'warden'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'wasabix'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'jetswap-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'Chain-2'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'yaxis'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'youswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'yuzuswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'zkswap'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'growth-defi'] = None

    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'ape-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'cache-gold'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'dot-dot-finance'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'dogewhale'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'dotoracle'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'ideamarket'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'lend-flare-dao-token'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'revault-network'] = None
    yes_gecko['gecko_id'][yes_gecko['gecko_id'] == 'moonswap'] = None


    yes_gecko['address'][yes_gecko['gecko_id'] == 'xy-finance'] = '0x77777777772cf0455fb38ee0e75f38034dfa50de'
    no_gecko['symbol'][no_gecko['symbol'] == 'Base'] = 'BASE'
    no_gecko['symbol'][no_gecko['symbol'] == 'PUZZLESWAP'] = 'PUZZLE'
    no_gecko['symbol'][no_gecko['symbol'] == 'safETH'] = 'SAFETH'

    no_gecko['gecko_id'][no_gecko['symbol'] == 'AAVE'] = 'aave'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ABC'] = 'abc-pos-pool'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ACA'] = 'acala'               # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'AGI'] = 'auragi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ALPACA'] = 'alpaca-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'AQUA'] = 'planet-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ASX'] = 'asymetrix'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BAL'] = 'balancer'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BASE'] = 'base'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BANANA'] = 'apeswap-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BEND'] = 'benddao'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BLUR'] = 'blur'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNB'] = 'binancecoin'         # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNC'] = 'bifrost-native-coin'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BNT'] = 'bancor'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BOW'] = 'archerswap-bow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CAKE'] = 'pancakeswap-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CAP'] = 'cap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CNC'] = 'conic-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'COMP'] = 'compound-governance-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CREAM'] = 'cream-2'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CRV'] = 'curve-dao-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DCHF'] = 'defi-franc'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DFI'] = 'defichain'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'EMPIRE'] = 'empire-network'   # CHAIN
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FLDX'] = 'flair-dex'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FLOW'] = 'velocimeter-flow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FOREX'] = 'handle-fi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FXS'] = 'frax-share'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GAMMA'] = 'green-planet'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GRAIL'] = 'camelot-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'HEC'] = 'hector-dao'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'HFT'] = 'hashflow'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'INTR'] = 'interlay'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'INV'] = 'inverse-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'IZI'] = 'izumi-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JBX'] = 'juicebox'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JOE'] = 'joe'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JST'] = 'just'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KLEVA'] = 'kleva'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KNC'] = 'kyber-network-crystal'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KSWAP'] = 'kyotoswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LIF3'] = 'lif3'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LSD'] = 'lsdx-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LVL'] = 'level'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MARE'] = 'mare-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MEGA'] = 'megaton-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MER'] = 'mercurial'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MET'] = 'metronome'
    no_gecko['gecko_id'][no_gecko['slug'] == 'mm-finance-arbitrum'] = 'mmfinance-arbitrum'
    no_gecko['gecko_id'][no_gecko['slug'] == 'mm-finance-cronos'] = 'mmfinance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MMO'] = 'mad-meerkat-optimizer'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MORPHO'] = 'morpho'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MPX'] = 'mpx'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MTA'] = 'meta'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'NEST'] = 'nest'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'NPT'] = 'neopin'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'OREO'] = 'oreoswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PLY'] = 'plenty-ply'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'QUICK'] = 'quickswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RBN'] = 'ribbon-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RDNT'] = 'radiant-capital'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RF'] = 'reactorfusion'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SAPR'] = 'swaprum'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SDL'] = 'stake-link'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SEAN'] = 'starfish-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SNEK'] = 'solisnek'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SOV'] = 'sovryn'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SPA'] = 'sperax'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SPIRIT'] = 'spiritswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'STELLA'] = 'stellaswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SUN'] = 'sun-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SUSHI'] = 'sushi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TETHYS'] = 'tethys-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TETU'] = 'tetu'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'THE'] = 'thena'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'THL'] = 'thala'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TOMB'] = 'tomb'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UMEE'] = 'umee'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UNI'] = 'uniswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VARA'] = 'equilibre'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VC'] = 'velocore'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VOLT'] = 'voltswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'WOO'] = 'woo-network'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'WYND'] = 'wynd'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'Y2K'] = 'y2k'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'YOK'] = 'yokaiswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ZYB'] = 'zyberswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'aUSD'] = 'acala-dollar-acala'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'bLUSD'] = 'boosted-lusd'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'csMatic'] = 'claystack-staked-matic'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'frxETH'] = 'frax-ether'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'kDAO'] = 'kolibri-dao'

    # NEW TOKENS
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ARBS'] = 'arbswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'EQ'] = 'equilibrium-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GGP'] = 'gogopool'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KOKOS'] = 'kokonut-swap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KTC'] = 'ktx-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LIT'] = 'timeless'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MAIA'] = 'maia'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MNGO'] = 'mango-markets'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PAL'] = 'paladin'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PIKA'] = 'pika-protocol'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PLSX'] = 'pulsex'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'QUIPU'] = 'quipuswap-governance-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RAM'] = 'ramses-exchange'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RCKT'] = 'rocketswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SCALE'] = 'scaleton'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'STONE'] = 'pangea-governance-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SUDO'] = 'sudoswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SYNTH'] = 'synthswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UNFI'] = 'unifi-protocol-dao'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VYFI'] = 'vyfinance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'XWIN'] = 'xwin-finance'

    # EXTRA TOKENS
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ALPHA'] = 'alpha-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BSW'] = 'biswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CHESS'] = 'tranchess'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CHR'] = 'chronos-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ARX'] = 'arbitrum-exchange'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BTRFLY'] = 'redacted'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CLA'] = 'claimswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DAO'] = 'dao-maker'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'FISH'] = 'polycat-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GLINT'] = 'beamswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MCB'] = 'mcdex'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MPL'] = 'maple'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PSYS'] = 'pegasys'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SRM'] = 'serum'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'STEAK'] = 'steakhut-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VALUE'] = 'value-liquidity'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'YAK'] = 'yield-yak'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ABEL'] = 'abel-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ALB'] = 'alienbase'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ATE'] = 'autoearn-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BASO'] = 'baso-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'BELA'] = 'beluga-protocol'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CANDY'] = 'bored-candy-city'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CHIRP'] = 'chirp-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CRISP-M'] = 'crisp-scored-mangroves'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'CTX'] = 'cryptex-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'DOV'] = 'doveswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'EAZY'] = 'eazyswap-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ECP'] = 'echodex-community-portion'
    no_gecko['gecko_id'][no_gecko['slug'] == 'el-dorado-exchange'] = 'el-dorado-exchange'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'EQUAL'] = 'equalizer-dex'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GHA'] = 'ghast'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GHNY'] = 'grizzly-honey'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'GMX'] = 'gmx'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'HEBE'] = 'hebeblock'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'JET'] = 'jet'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'KUJI'] = 'kujira'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LFI'] = 'lunafi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'LODE'] = 'lodestar'
    no_gecko['gecko_id'][no_gecko['slug'] == 'meta-pool-eth'] = 'meta-pool'
    no_gecko['gecko_id'][no_gecko['slug'] == 'meta-pool-near'] = 'meta-pool'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'MYC'] = 'mycelium'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'NUON'] = 'nuon'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ON'] = 'aegis-token-f7934368-2fb3-4091-9edc-39283e87f55d'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ORACLE'] = 'oracleswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PARA'] = 'parallel-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PIKO'] = 'pinnako'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PINKAV'] = 'pinjam-kava'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PRTC'] = 'protectorate-protocol'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'PUZZLE'] = 'puzzle-swap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RADT'] = 'radiate-protocol'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'RUBY'] = 'ruby'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SKCS'] = 'staked-kcs'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'STSW'] = 'stackswap'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SVY'] = 'savvy-defi'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SWORD'] = 'ezkalibur'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'TITI'] = 'titi-governance-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UNCX'] = 'unicrypt-2'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'UNION'] = 'union-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'VELO'] = 'velodrome-finance'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'WHALE'] = 'white-whale'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'XVS'] = 'venus'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'ZLK'] = 'zenlink-network-token'
    no_gecko['gecko_id'][no_gecko['symbol'] == 'SAFETH'] = 'simple-asymmetry-eth'

    no_gecko['id_collection'] = np.where(no_gecko['gecko_id'].isna(), 'None', 'Manually')
    yes_gecko['id_collection'] = np.where(yes_gecko['gecko_id'].isna(), 'Manually', 'Automatically')
    llama_data = pd.concat([no_gecko, yes_gecko]).sort_values('symbol').reset_index(drop = True)

    # chain_data = pd.DataFrame(columns = llama_data.columns)
    # chain_data.loc[0] = ['Avalanche', 'avax:0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7', 'AVAX', 'Avalanche', None, 'avalanche-2', 'Other', None, 'Manually']
    # chain_data.loc[1] = ['BNB', 'bsc:0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c', 'BNB', 'Binance', None, 'binancecoin', 'Other', None, 'Manually']
    # chain_data.loc[2] = ['Ethereum', '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'ETH', 'Ethereum', None, 'ethereum', 'Other', None, 'Manually']
    # chain_data.loc[3] = ['Polygon', 'polygon:0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270', 'MATIC', 'Polygon', None, 'matic-network', 'Other', None, 'Manually']
    # llama_data = pd.concat([llama_data, chain_data]).sort_values('symbol')
    # llama_data = llama_data.drop_duplicates().reset_index(drop = True)

    return llama_data