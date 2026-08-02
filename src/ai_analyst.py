from typing import Dict, Optional
import httpx

class AIAnalyst:
    def __init__(self, api_key: str, base_url: str = "https://routerai.ru/api/v1", model: str = "deepseek/deepseek-v4-pro", timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
    
    def analyze(self, metrics, strategy_name, symbol, interval, language="ru"):
        prompt = self._build_prompt(metrics, strategy_name, symbol, interval, language)
        response = self._call_api(prompt)
        return response if response else self._fallback_analysis(metrics)
    
    def _build_prompt(self, metrics, strategy_name, symbol, interval, language):
        lang = "Отвечай на русском языке." if language == "ru" else "Answer in English."
        return f"""Ты — профессиональный квант-аналитик и риск-менеджер криптовалютного хедж-фонда.
Проанализируй результаты бэктеста и дай подробные рекомендации.
Стратегия: {strategy_name} | Пара: {symbol} | Интервал: {interval}
Результаты:
- Доходность: {metrics.get('total_return_pct', 0):.2f}%
- Общая прибыль: ${metrics.get('total_pnl', 0):,.2f}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
- Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
- Макс. просадка: {metrics.get('max_drawdown_pct', 0):.2f}%
- Годовая волатильность: {metrics.get('annual_volatility', 0):.2f}%
- Win Rate: {metrics.get('win_rate', 0):.1f}%
- Profit Factor: {metrics.get('profit_factor', 0):.2f}
- Всего сделок: {metrics.get('total_trades', 0)}
- Средний выигрыш: ${metrics.get('avg_win', 0):,.2f}
- Средний проигрыш: ${metrics.get('avg_loss', 0):,.2f}
Задачи: 1) Оцени качество стратегии. 2) Выдели риски. 3) Дай рекомендации по запуску. 4) Какие рыночные условия опасны?
{lang} Формат: структурированный текст с эмодзи-заголовками, без markdown. Будь конкретным и честным."""
    
    def _call_api(self, prompt):
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/chat/completions",
                    headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"},
                    json={"model":self.model,"messages":[{"role":"system","content":"Ты эксперт по криптотрейдингу."},{"role":"user","content":prompt}],"temperature":0.7,"max_tokens":1200})
                if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
                else: print(f"API error: {r.status_code}"); return None
        except Exception as e: print(f"API call failed: {e}"); return None
    
    def _fallback_analysis(self, metrics):
        sharpe, dd, pf, wr, ret, trades = metrics.get('sharpe_ratio',0), abs(metrics.get('max_drawdown_pct',0)), metrics.get('profit_factor',0), metrics.get('win_rate',0), metrics.get('total_return_pct',0), metrics.get('total_trades',0)
        if sharpe > 2.0 and dd < 15 and pf > 2.0 and trades > 10: rating, action, risk = "🟢 ОТЛИЧНАЯ стратегия", "Можно в реальную торговлю с 25% депозита.", "НИЗКИЙ"
        elif sharpe > 1.0 and dd < 25 and pf > 1.3: rating, action, risk = "🟡 ХОРОШАЯ стратегия", "Демо-счёт 2-4 недели, затем 10-15% капитала.", "СРЕДНИЙ"
        elif sharpe > 0.3 and dd < 35 and pf > 1.0: rating, action, risk = "🟠 СРЕДНЯЯ стратегия", "Требуется доработка. Только демо-счёт.", "ПОВЫШЕННЫЙ"
        elif trades < 5: rating, action, risk = "⚪ НЕДОСТАТОЧНО ДАННЫХ", "Увеличьте период бэктеста.", "НЕ ОПРЕДЕЛЁН"
        else: rating, action, risk = "🔴 СЛАБАЯ стратегия", "Не рекомендуется к торговле.", "ВЫСОКИЙ"
        return f"""ОЦЕНКА: {rating}
📊 КЛЮЧЕВЫЕ МЕТРИКИ:
• Доходность: {ret:.2f}% • Sharpe: {sharpe:.2f} • Макс. просадка: {dd:.2f}% • Win Rate: {wr:.1f}% • Profit Factor: {pf:.2f} • Сделок: {trades}
⚠️ АНАЛИЗ РИСКОВ:
• Уровень риска: {risk}
{f"• Высокая просадка ({dd:.1f}%)" if dd>25 else f"• Просадка ({dd:.1f}%) в пределах нормы"}
{f"• Низкий Sharpe ({sharpe:.2f})" if sharpe<0.5 else f"• Sharpe ({sharpe:.2f}) приемлемый"}
{f"• Низкий Win Rate ({wr:.1f}%)" if wr<40 else f"• Win Rate ({wr:.1f}%) комфортный"}
💡 РЕКОМЕНДАЦИЯ: {action}
⚠️ ВАЖНО: AI-запрос не удался. Показан офлайн-анализ. Проверьте ключ, модель и Base URL."""