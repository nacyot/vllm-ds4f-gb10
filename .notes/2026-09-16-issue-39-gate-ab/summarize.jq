# jq -s -f summarize.jq casebench.jsonl decode.jsonl
["i39-off", "i39-pf", "i39-split", "i39-prod"] as $tags |
["warm", "warm2", "S128r1", "S128r2", "S128r3", "S32", "S128r4"] as $cells |
. as $rows |
[$rows[] | select(.config != null)] as $cases |
[$rows[] | select(.type != null)] as $decode |
if ($cases | length) != 28 or ($decode | length) != 24 then
  error("expected casebench 28 and decode 24 rows")
else . end |
[$tags[] as $tag |
  [$cases[] | select(.config == $tag)] as $c |
  [$decode[] | select(.tag == $tag)] as $d |
  if ($c | map(.tag) | sort) != ($cells | map($tag + "-" + .) | sort)
    or ($d | map([.type, .c]) | sort) !=
       (["prose", "code"] | [.[] as $kind | [1, 2, 4][] | [$kind, .]] | sort)
  then error("missing or duplicate cells: " + $tag) else . end |
  if (all($c[]; (.prefills | length) == 1 and
      .prefills[0].prompt_tokens > 0 and .aggregate.tok_per_s > 0) | not)
    or (all($d[]; .streams_ok == .c and .errors == [] and
      .gen == 256 and (.completion_tokens | length) == .c and
      all(.completion_tokens[]; . == 256) and
      .per_stream_1 > 0 and .tokens_per_chunk > 0) | not)
  then error("unsuccessful benchmark output: " + $tag) else . end |
  [range(1; 5) as $i | $c[] |
    select(.tag == ($tag + "-S128r" + ($i | tostring))) |
    .aggregate.tok_per_s] as $s |
  ($d[] | select(.type == "code" and .c == 1)) as $code |
  {
    tag: $tag,
    s128: $s,
    mean: (($s[1] + $s[2]) / 2),
    order: ($s[3] - (($s[1] + $s[2]) / 2)),
    s32: ($c[] | select(.tag == ($tag + "-S32")) | .aggregate.tok_per_s),
    codeStep: (1000 * $code.tokens_per_chunk / $code.per_stream_1),
    proseC1: ($d[] | select(.type == "prose" and .c == 1) | .per_stream_1),
    memMin: ($c | map(.head_mem.min_gib) | min)
  }
]
