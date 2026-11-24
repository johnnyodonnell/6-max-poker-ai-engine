// pm2 config for the engine service on the box. From
// ~/Workspace/6-max-poker-ai-engine/repo:
//
//   pm2 start cfr/pm2.config.cjs   # start (or: pm2 restart sixmax-engine)
//   pm2 save                       # bring it back after a reboot
//   pm2 logs sixmax-engine
//
// Logs go to the project folder; only pm2's own bookkeeping lives in ~/.pm2.
const SIXMAX_HOME = `${process.env.HOME}/Workspace/6-max-poker-ai-engine`;

module.exports = {
  apps: [
    {
      name: "sixmax-engine",
      cwd: `${SIXMAX_HOME}/repo/cfr`,
      script: "./serve.sh",
      interpreter: "bash",
      env: { SIXMAX_HOME },
      out_file: `${SIXMAX_HOME}/logs/engine.log`,
      error_file: `${SIXMAX_HOME}/logs/engine.err.log`,
      time: true,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
    },
  ],
};
