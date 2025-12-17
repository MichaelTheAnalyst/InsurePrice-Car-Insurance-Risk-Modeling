variable "prefix" {
  description = "The prefix which should be used for all resources in this example"
  default     = "insureprice"
}

variable "location" {
  description = "The Azure Region in which all resources in this example should be created."
  default     = "northeurope"
}

variable "image_name" {
  description = "The name of the docker image"
  default     = "insureprice"
}
