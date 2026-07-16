const GenreBadge = ({ genre }: { genre: string }) => {
  const colors: Record<string, string> = {
    Rock: 'bg-red-500/10 text-red-400 border-red-500/20',
    Pop: 'bg-pink-500/10 text-pink-400 border-pink-500/20',
    Jazz: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    Electronic: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    'Hip-Hop': 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    Classical: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    'R&B': 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    Country: 'bg-amber-600/10 text-amber-500 border-amber-600/20',
    Metal: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
    Reggae: 'bg-green-500/10 text-green-400 border-green-500/20',
    Indie: 'bg-teal-500/10 text-teal-400 border-teal-500/20',
    Blues: 'bg-blue-600/10 text-blue-400 border-blue-600/20',
    Folk: 'bg-stone-500/10 text-stone-400 border-stone-500/20',
    Latin: 'bg-red-600/10 text-red-400 border-red-600/20',
    Soul: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
    Funk: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    Punk: 'bg-rose-600/10 text-rose-400 border-rose-600/20',
    Ambient: 'bg-sky-400/10 text-sky-400 border-sky-400/20',
    'Pop Rock': 'bg-orange-400/10 text-orange-400 border-orange-400/20',
    Alternative: 'bg-lime-500/10 text-lime-400 border-lime-500/20',
  }
  const styling = colors[genre] ?? 'bg-gray-500/10 text-gray-400 border-gray-500/20'

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider border ${styling}`}>
      {genre}
    </span>
  )
}

export default GenreBadge
