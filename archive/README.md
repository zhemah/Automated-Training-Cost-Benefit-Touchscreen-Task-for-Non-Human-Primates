# Archive

Nothing in this folder needs running. It is kept as a record of something that was
done once and is now finished.

## `migrate_to_current_protocol.py`

When the task went from two shapes to four, animals that were already part-way
through training had records written under the older version. Their files said
"Phase 5", but Phase 5 meant something different then, so simply carrying on would
have mixed old and new sessions together and confused the checks that decide when an
animal is ready to move up.

This script read an animal's history to work out which shapes they had actually seen
and in what order, then worked out where they should restart under the new scheme.
Where the old and new records would have overlapped, it added a single marker row so
the program would begin a clean session instead of continuing an old one. It never
edited or deleted anything that was already there.

**It has already been run on every animal in the colony, and no future animal will
need it.** A monkey starting today begins on the current version, so there is no
older version to be brought forward from.

It is kept because it documents a change that was made to real records. If anyone
later asks why one animal's shape order differs from the others, or why there is a
jump in session numbers partway through a file, the answer is here.
