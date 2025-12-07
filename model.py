import shift
from time import sleep
from datetime import timedelta
from threading import Thread

import numpy as np
from arch import arch_model
from collections import deque

###############################################################################
# ------------------------ CONFIG / PARAMETERS -------------------------------
###############################################################################

# Tickers to trade (each ticker runs in its own thread)
SYMBOLS = ["AAPL", "AMZN", "BRKb", "GOOG", "MSFT"]

TRADE_DURATION_MINUTES = 10

# GARCH window
PRICE_WINDOW = 120   # number of mid-prices to keep

# Avellaneda–Stoikov parameters
gamma = 0.01         # risk aversion
kappa = 1.5          # market depth
phi_max = 2          # max size (in lots)
eta = -0.005         # inventory shape

###############################################################################
# ------------------------ GLOBAL STATE --------------------------------------
###############################################################################

rolling_mids = {sym: deque(maxlen=PRICE_WINDOW) for sym in SYMBOLS}
current_sigma = {sym: 0.50 for sym in SYMBOLS}   # fallback sigma until GARCH updates
last_valid_mid = {sym: None for sym in SYMBOLS}

###############################################################################
# ------------------------ HELPERS -------------------------------------------
###############################################################################

def get_safe_mid(best, sym):
    """
    Use midprice only, but handle cases where bid/ask are missing.
    """
    bid = best.get_bid_price()
    ask = best.get_ask_price()

    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        last_valid_mid[sym] = mid
        return mid

    if bid > 0:
        last_valid_mid[sym] = bid
        return bid

    if ask > 0:
        last_valid_mid[sym] = ask
        return ask

    if last_valid_mid[sym] is not None:
        return last_valid_mid[sym]

    # absolute fallback
    print(f"[WARN] No valid mid for {sym}, using 0.01")
    return 0.01


def estimate_sigma_garch(returns):
    """
    GARCH(1,1) on log returns, with mild scaling and safe fallback.
    """
    returns = np.array(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    returns = returns[np.abs(returns) > 1e-12]

    if len(returns) < 10:
        return 0.50

    # mild scaling for numerical stability
    scaled = returns * 100.0

    try:
        am = arch_model(
            scaled,
            vol="Garch",
            p=1, q=1,
            mean="Zero",
            dist="normal",
            rescale=False,
        )
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1)
        var_t = forecast.variance.iloc[-1, 0]

        sigma_scaled = float(np.sqrt(var_t))
        sigma_t = sigma_scaled / 100.0  # undo scaling

        # clamp to reasonable range
        sigma_t = max(1e-4, min(sigma_t, 1.0))
        return sigma_t

    except Exception as e:
        print(f"[GARCH ERROR] {e}")
        if len(returns) > 1:
            sigma_t = returns.std()
            sigma_t = max(1e-4, min(float(sigma_t), 1.0))
            return sigma_t
        return 0.50


def compute_dynamic_size(q):
    """
    Inventory-dependent size from AS-style φ_bid, φ_ask.
    q is inventory in shares.
    """
    if q < 0:  # short → buy aggressively
        phi_bid = phi_max
        phi_ask = phi_max * np.exp(-eta * q)
    else:      # long → sell aggressively
        phi_bid = phi_max * np.exp(-eta * q)
        phi_ask = phi_max
    return max(1, int(phi_bid)), max(1, int(phi_ask))


def compute_as_quotes(mid, q, sigma, gamma, kappa, T_minus_t):
    """
    Pure Avellaneda–Stoikov quoting:

    r_t = s_t − q_t γ σ² (T − t)
    δ_a + δ_b = γ σ² (T − t) + 2 ln(1 + γ/κ)
    bid = r_t − δ, ask = r_t + δ
    """
    r = mid - q * gamma * (sigma ** 2) * T_minus_t
    spread = gamma * (sigma ** 2) * T_minus_t + 2.0 * np.log(1.0 + gamma / kappa)
    delta = spread / 2.0

    bid = round(r - delta, 2)
    ask = round(r + delta, 2)

    if bid >= ask:
        bid = ask - 0.01

    bid = max(0.01, bid)
    return bid, ask


def cancel_orders(trader, ticker):
    """
    Cancel all remaining orders for a specific ticker.
    """
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            trader.submit_cancellation(order)
            sleep(0.25)


def close_positions(trader, ticker):
    """
    Close all long/short positions for the given ticker using market orders.
    """
    print(f"[CLOSE POSITIONS] {ticker}")

    item = trader.get_portfolio_item(ticker)

    # close any long positions
    long_shares = item.get_long_shares()
    if long_shares > 0:
        print(f"  market selling {ticker}, long shares = {long_shares}")
        order = shift.Order(
            shift.Order.Type.MARKET_SELL,
            ticker,
            long_shares // 100,  # lots
        )
        trader.submit_order(order)
        sleep(0.5)

    # close any short positions
    short_shares = item.get_short_shares()
    if short_shares > 0:
        print(f"  market buying {ticker}, short shares = {short_shares}")
        order = shift.Order(
            shift.Order.Type.MARKET_BUY,
            ticker,
            short_shares // 100,
        )
        trader.submit_order(order)
        sleep(0.5)

###############################################################################
# ------------------------ STRATEGY (PER THREAD) -----------------------------
###############################################################################

def strategy(trader: shift.Trader, ticker: str, endtime):
    """
    Threaded strategy function for a single ticker.
    Runs AS + GARCH until endtime, then closes positions.
    """
    print(f"[START STRATEGY] {ticker}")

    check_freq = 1.0  # seconds between loops
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()

    # initialize previous mid
    best_price = trader.get_best_price(ticker)
    prev_mid = get_safe_mid(best_price, ticker)

    while trader.get_last_trade_time() < endtime:
        # cancel any resting orders for this ticker
        cancel_orders(trader, ticker)

        # get updated best prices and mid
        best_price = trader.get_best_price(ticker)
        mid = get_safe_mid(best_price, ticker)

        # push mid into rolling window
        rolling_mids[ticker].append(mid)
        mids = np.array(rolling_mids[ticker], dtype=float)
        mids = mids[mids > 0]

        # compute log-returns for GARCH
        if len(mids) > 1:
            logp = np.log(mids)
            rets = np.diff(logp)
        else:
            rets = np.array([])

        # update sigma once we have enough data
        if len(mids) >= 20 and len(rets) >= 5:
            sigma_new = estimate_sigma_garch(rets)
            current_sigma[ticker] = sigma_new

        sigma = current_sigma[ticker]

        # inventory in shares
        q = trader.get_portfolio_item(ticker).get_shares()

        # local AS horizon (seconds) – simple constant
        T_minus_t = 30.0

        # compute AS quotes
        bid_px, ask_px = compute_as_quotes(mid, q, sigma, gamma, kappa, T_minus_t)
        size_bid, size_ask = compute_dynamic_size(q)

        # cap max size (lots)
        size_bid = min(size_bid, 5)
        size_ask = min(size_ask, 5)

        print(
            f"[{ticker}] mid={mid:.4f} q={q} σ={sigma:.6f} "
            f"BID={bid_px}x{size_bid}  ASK={ask_px}x{size_ask}"
        )

        # place new AS quotes
        trader.submit_order(shift.Order(shift.Order.Type.LIMIT_BUY, ticker, size_bid, bid_px))
        trader.submit_order(shift.Order(shift.Order.Type.LIMIT_SELL, ticker, size_ask, ask_px))

        prev_mid = mid
        sleep(check_freq)

    # after loop: cancel and close
    cancel_orders(trader, ticker)
    close_positions(trader, ticker)

    final_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    print(f"[RESULT] {ticker} P&L = {final_pl - initial_pl:.2f}")



def main(trader: shift.Trader):
    """
    Leaderboard-compatible main function.
    Sets the session start/end, launches one thread per ticker, and waits.
    """
    # reference "current" simulation time from SHIFT
    current = trader.get_last_trade_time()

    start_time = current
    end_time = start_time + timedelta(minutes=TRADE_DURATION_MINUTES)

    print(f"[CONFIG] AS-GARCH strategy for {TRADE_DURATION_MINUTES} minutes")
    print(f"[TIME] start={start_time}  end={end_time}")

    # wait until start_time (for completeness; usually immediate)
    while trader.get_last_trade_time() < start_time:
        print("still waiting for market open...")
        sleep(1)

    threads = []
    tickers = SYMBOLS

    print("[START SIMULATION]")

    for ticker in tickers:
        t = Thread(target=strategy, args=(trader, ticker, end_time))
        threads.append(t)

    # start each thread with a short stagger
    for t in threads:
        t.start()
        sleep(0.25)

    # wait until end_time
    while trader.get_last_trade_time() < end_time:
        sleep(1)

    # wait for all strategy threads to finish
    for t in threads:
        t.join()

    # final safety: ensure all positions closed
    for ticker in tickers:
        cancel_orders(trader, ticker)
        close_positions(trader, ticker)

    print("[END SIMULATION]")
    print(f"final bp:  {trader.get_portfolio_summary().get_total_bp()}")
    print(f"final P&L: {trader.get_portfolio_summary().get_total_realized_pl():.2f}")


###############################################################################
# ------------------------ ENTRYPOINT ----------------------------------------
###############################################################################

if __name__ == '__main__':
    with shift.Trader("") as trader:
        trader.connect("initiator.cfg", "")
        sleep(1)

        trader.sub_all_order_book()
        sleep(1)

        main(trader)
