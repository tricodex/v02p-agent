from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import logging
from cdp import Wallet
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioConfig(BaseModel):
    """Portfolio configuration."""
    target_assets: List[str]
    target_allocation: Dict[str, float]
    rebalancing_threshold: float = Field(default=0.02, ge=0, le=1)
    max_single_trade_size: Decimal = Field(default=Decimal("10000"))
    auto_rebalancing: bool = Field(default=True)

class PortfolioManager:
    """Portfolio management using CDP AgentKit."""
    
    def __init__(
        self,
        cdp_wrapper: CdpAgentkitWrapper,
        config: PortfolioConfig,
        network_id: str = "base-sepolia"
    ):
        self.cdp_wrapper = cdp_wrapper
        self.config = config
        self.network_id = network_id
        self.wallet = self.cdp_wrapper.wallet
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp_wrapper)
        
    async def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        try:
            balances = {}
            total_value = Decimal(0)
            
            # Get balances for supported assets
            for asset_id in self.config.target_assets:
                balance = await self.wallet.balance(asset_id.lower())
                price_data = await self._get_price(asset_id)
                value = balance * price_data["price"]
                
                balances[asset_id] = {
                    "balance": str(balance),
                    "price": str(price_data["price"]),
                    "value_usd": str(value),
                    "change_24h": price_data.get("change_24h", 0)
                }
                total_value += value
            
            # Calculate allocations
            allocations = {
                asset_id: float(Decimal(data["value_usd"]) / total_value)
                for asset_id, data in balances.items()
                if total_value > 0
            }
            
            return {
                "total_value_usd": str(total_value),
                "asset_allocation": allocations,
                "balances": balances,
                "last_updated": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {str(e)}")
            raise
    
    async def execute_trade(
        self,
        asset_id: str,
        amount: str,
        side: str,
        max_slippage: float = 0.01
    ) -> Dict:
        """Execute a trade using CDP AgentKit."""
        try:
            # Execute trade
            if side.lower() == "buy":
                trade = await self.wallet.trade(
                    amount=amount,
                    from_asset_id="usdc",
                    to_asset_id=asset_id.lower()
                )
            else:
                trade = await self.wallet.trade(
                    amount=amount,
                    from_asset_id=asset_id.lower(),
                    to_asset_id="usdc"
                )
            
            # Wait for confirmation
            result = await trade.wait()
            
            return {
                "status": "completed",
                "transaction_hash": result.transaction.transaction_hash,
                "transaction_link": result.transaction.transaction_link,
                "execution_price": str(result.execution_price) if result.execution_price else None,
                "gas_used": str(result.gas_used) if result.gas_used else None
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def rebalance_portfolio(
        self,
        max_slippage: float = 0.01
    ) -> List[Dict]:
        """Rebalance portfolio to target allocation."""
        try:
            stats = await self.get_portfolio_stats()
            current_allocation = stats["asset_allocation"]
            total_value = Decimal(stats["total_value_usd"])
            
            trades = []
            for asset_id, target_weight in self.config.target_allocation.items():
                current_weight = current_allocation.get(asset_id, 0)
                if abs(current_weight - target_weight) > self.config.rebalancing_threshold:
                    # Calculate trade size
                    trade_value = abs(target_weight - current_weight) * total_value
                    
                    # Split into smaller trades if needed
                    if trade_value > self.config.max_single_trade_size:
                        n_trades = int((trade_value / self.config.max_single_trade_size).ceil())
                        trade_value = trade_value / n_trades
                    else:
                        n_trades = 1
                    
                    # Execute trades
                    for _ in range(n_trades):
                        result = await self.execute_trade(
                            asset_id=asset_id,
                            amount=str(trade_value),
                            side="buy" if target_weight > current_weight else "sell",
                            max_slippage=max_slippage
                        )
                        trades.append(result)
            
            return trades
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {str(e)}")
            raise
    
    async def _get_price(self, asset_id: str) -> Dict:
        """Get current price for an asset."""
        # This is a mock implementation
        # In production, integrate with a price feed service
        mock_prices = {
            "ETH": {"price": Decimal("2000"), "change_24h": 5.2},
            "BTC": {"price": Decimal("40000"), "change_24h": 3.1},
            "USDC": {"price": Decimal("1"), "change_24h": 0.0}
        }
        return mock_prices.get(asset_id.upper(), {"price": Decimal("0"), "change_24h": 0})

# Usage example:
async def main():
    # Initialize CDP wrapper
    cdp_wrapper = CdpAgentkitWrapper(
        network_id="base-sepolia",
        cdp_api_key_name="your-api-key-name",
        cdp_api_key_private_key="your-api-key-private-key"
    )
    
    # Create portfolio config
    config = PortfolioConfig(
        target_assets=["ETH", "USDC"],
        target_allocation={"ETH": 0.5, "USDC": 0.5},
        rebalancing_threshold=0.02,
        max_single_trade_size=Decimal("1000"),
        auto_rebalancing=True
    )
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(cdp_wrapper, config)
    
    # Get portfolio stats
    stats = await portfolio_manager.get_portfolio_stats()
    print(f"Portfolio value: ${stats['total_value_usd']}")
    print(f"Current allocation: {stats['asset_allocation']}")
    
    # Rebalance if needed
    if config.auto_rebalancing:
        trades = await portfolio_manager.rebalance_portfolio()
        print(f"Executed trades: {trades}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())