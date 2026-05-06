//! Agent lifecycle and multi-agent communication methods.
//!
//! SubAgent, NotifyParent, MessageAgent, BroadcastAgents, ListAgents,
//! GetAgent, TransferInput — all extracted from the main `ToolExecutor` impl.

use serde::Deserialize;
use anyhow::Context as _;
use uuid::Uuid;

use crate::cancel::CancelToken;
use super::{
    ToolExecutor, ToolRuntime, AgentMessageRequest, AgentScope,
    BroadcastAgentsRequest, ListAgentsRequest,
    SubAgentRequest, TransferInputRequest,
};

impl ToolExecutor {
    /// `SubAgent`: delegate a focused sub-task to a nested agent.
    pub(crate) async fn subagent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("SubAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            #[serde(default, deserialize_with = "super::deserialize_optional_nonempty_string")]
            label: Option<String>,
            task: String,
            context: String,
            #[serde(default, deserialize_with = "super::deserialize_optional_nonempty_string")]
            prompt: Option<String>,
            #[serde(default, deserialize_with = "super::deserialize_optional_nonempty_string")]
            prompt_file: Option<String>,
            #[serde(default, deserialize_with = "super::deserialize_optional_nonempty_string")]
            prompt_skill: Option<String>,
            #[serde(default)]
            allow_user_send: bool,
            #[serde(default)]
            allow_user_show: bool,
            #[serde(default)]
            allow_user_ask: bool,
            #[serde(default)]
            allow_input_transfer_target: bool,
            #[serde(default, deserialize_with = "super::deserialize_optional_uuid_or_empty")]
            existing_agent_id: Option<Uuid>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for SubAgent")?;
        let request = SubAgentRequest {
            label: args.label,
            task: args.task,
            context: args.context,
            prompt: args.prompt,
            prompt_file: args.prompt_file,
            prompt_skill: args.prompt_skill,
            allow_user_send: args.allow_user_send,
            allow_user_show: args.allow_user_show,
            allow_user_ask: args.allow_user_ask,
            allow_input_transfer_target: args.allow_input_transfer_target,
            existing_agent_id: args.existing_agent_id,
        };
        request.validate()?;

        let handle = runtime.run_subagent(request, cancel.clone()).await?;

        Ok(serde_json::json!({
            "agent_id": handle.agent_id,
            "work_id": handle.work_id,
            "label": handle.label,
            "status": handle.status,
        })
        .to_string())
    }

    /// `NotifyParent`: convenience one-way message to the current parent.
    pub(crate) async fn notify_parent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("NotifyParent cancelled");
        }
        if runtime.parent_agent_id.is_none() {
            anyhow::bail!("NotifyParent is only available for child agents");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for NotifyParent")?;
        if args.message.trim().is_empty() {
            anyhow::bail!("NotifyParent message must not be empty");
        }

        let receipt = runtime.notify_parent(args.message.clone()).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `MessageAgent`: direct one-way message to another agent.
    pub(crate) async fn message_agent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("MessageAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            target_agent_id: Uuid,
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MessageAgent")?;
        let request = AgentMessageRequest {
            target_agent_id: args.target_agent_id,
            message: args.message,
        };
        request.validate()?;
        let receipt = runtime.message_agent(request).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `BroadcastAgents`: broadcast a one-way message to a scoped set of agents.
    pub(crate) async fn broadcast_agents(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("BroadcastAgents cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            scope: Option<AgentScope>,
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for BroadcastAgents")?;
        let request = BroadcastAgentsRequest {
            scope: args.scope.unwrap_or(AgentScope::Descendants),
            message: args.message,
        };
        request.validate()?;
        let receipt = runtime.broadcast_agents(request).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `ListAgents`: inspect visible agents.
    pub(crate) async fn list_agents(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ListAgents cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            scope: Option<AgentScope>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ListAgents")?;
        let items = runtime
            .list_agents(ListAgentsRequest {
                scope: args.scope.unwrap_or(AgentScope::Descendants),
            })
            .await?;
        Ok(serde_json::to_string(&items)?)
    }

    /// `GetAgent`: inspect one visible agent.
    pub(crate) async fn get_agent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            agent_id: Uuid,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for GetAgent")?;
        let item = runtime.get_agent(args.agent_id).await?;
        Ok(serde_json::to_string(&item)?)
    }

    /// `TransferInput`: move free-form user input ownership.
    pub(crate) async fn transfer_input(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("TransferInput cancelled");
        }
        if !runtime.is_root {
            anyhow::bail!("TransferInput is only available to the root agent");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            target_agent_id: Option<Uuid>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for TransferInput")?;
        let receipt = runtime
            .transfer_input(TransferInputRequest {
                target_agent_id: args.target_agent_id,
            })
            .await?;
        Ok(serde_json::to_string(&receipt)?)
    }
}
