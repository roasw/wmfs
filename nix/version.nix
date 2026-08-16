{
  lib,
  self,
}:
let
  dirty = self ? dirtyRev && !(self ? rev);
  revision = lib.removeSuffix "-dirty" (self.rev or self.dirtyRev or "unknown");
  shortRevision = builtins.substring 0 12 revision;
  gitVersion = shortRevision + lib.optionalString dirty "-dirty";
in
{
  git = gitVersion;
  python = "0.0.0+g${shortRevision}${lib.optionalString dirty ".dirty"}";
}
