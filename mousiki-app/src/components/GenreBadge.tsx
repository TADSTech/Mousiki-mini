const GenreBadge = ({ genre }: { genre: string }) => {
  const colors: Record<string, string> = {
    Rock: 'bg-red-500/20 text-red-300',
    Pop: 'bg-pink-500/20 text-pink-300',
    Jazz: 'bg-amber-500/20 text-amber-300',
    Electronic: 'bg-cyan-500/20 text-cyan-300',
    'Hip-Hop': 'bg-yellow-500/20 text-yellow-300',
    Classical: 'bg-indigo-500/20 text-indigo-300',
    'R&B': 'bg-purple-500/20 text-purple-300',
    Country: 'bg-amber-600/20 text-amber-300',
    Metal: 'bg-gray-500/20 text-gray-300',
    Reggae: 'bg-green-500/20 text-green-300',
    Indie: 'bg-teal-500/20 text-teal-300',
    Blues: 'bg-blue-600/20 text-blue-300',
    Folk: 'bg-stone-500/20 text-stone-300',
    Latin: 'bg-red-600/20 text-red-300',
    Soul: 'bg-violet-500/20 text-violet-300',
    Funk: 'bg-orange-500/20 text-orange-300',
    Punk: 'bg-rose-600/20 text-rose-300',
    Ambient: 'bg-sky-400/20 text-sky-300',
    'Pop Rock': 'bg-orange-400/20 text-orange-300',
    Alternative: 'bg-lime-500/20 text-lime-300',
  }
  const bg = colors[genre] ?? 'bg-gray-500/20 text-gray-300'

  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${bg}`}>
      {genre}
    </span>
  )
}

export default GenreBadge
