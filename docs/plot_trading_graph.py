import numpy as np
import matplotlib.pyplot as plt


def plot_trading_pattern( midprice, bestbuy, bestsell, buyrate, sellrate, filename ):

    plt.figure(figsize=(18, 10), dpi=80)

    ax = plt.axes()
    ax.set_facecolor('whitesmoke')
    plt.xlabel( 'Step' )
    plt.ylabel( 'Log10 of Mid Price Ratio' )

    xvals = np.arange( len( midprice))
    plt.plot(    xvals, np.log10( bestbuy  / midprice), linewidth = 0.5, c='g' )
    plt.plot(    xvals, np.log10( bestsell / midprice), linewidth = 0.5, c='r' )
    plt.scatter( xvals, np.log10( buyrate  / midprice), s = 5, marker='s', c='black')
    plt.scatter( xvals, np.log10( sellrate / midprice), s = 5, marker='s', c='deepskyblue' )

    plt.savefig( filename )    



# Make Artificial Data --------
nsteps = 1000
tick = 0.00000001
fairprice = np.cumsum( np.random.normal( 0, 0.00001, nsteps)) + 0.001
bestbuy  = fairprice - tick + np.minimum ( 0, np.random.normal( -0.000001, 0.000001, nsteps ))
bestsell = fairprice + tick + np.maximum ( 0, np.random.normal(  0.000001, 0.000001, nsteps ))
midprice = ( bestbuy + bestsell ) / 2

buyrate  = midprice - tick + np.minimum ( 0, np.random.normal( -0.000001, 0.000001, nsteps ))
sellrate = midprice + tick + np.maximum ( 0, np.random.normal(  0.000001, 0.000001, nsteps ))
#------------------------------


plot_trading_pattern( midprice, bestbuy, bestsell, buyrate, sellrate, 'tradingplot.png' )
