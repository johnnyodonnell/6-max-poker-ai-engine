// One chip is one dollar (the game plays as $1/$2 with $200 stacks).
export function dollars(amount: number): string {
  return `$${amount.toLocaleString()}`;
}
